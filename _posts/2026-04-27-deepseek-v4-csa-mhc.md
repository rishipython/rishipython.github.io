---
layout: post
title: "deepseek-v4: how csa and mhc make 1m-token contexts cheap"
date: 2026-04-27
tags: [deepseek, attention, long-context, residual-connections, mixture-of-experts]
---

references: [deepseek-v4 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf){:target="_blank"}, [openlm.ai overview](https://openlm.ai/deepseek-v4/){:target="_blank"}, [mhc paper (xie et al., 2026)](https://arxiv.org/abs/2512.24880){:target="_blank"}

deepseek-v4 is a moe family (v4-pro at 1.6t total / 49b active, v4-flash at 284b total / 13b active) with native 1m-token context. the headline numbers are the efficiency ones: at 1m tokens, v4-pro uses ~27% of the per-token inference flops and ~10% of the kv cache of v3.2. those gains come from two architectural pieces worth understanding in detail:

1. a hybrid attention stack that interleaves **compressed sparse attention (csa)** and **heavily compressed attention (hca)**.
2. **manifold-constrained hyper-connections (mhc)** that replace standard residual connections to keep deep stacks numerically stable.

this post walks through csa and mhc with their actual math, since the design choices only really make sense once you see the equations.

## compressed sparse attention (csa)

vanilla self-attention is `O(n^2)` in sequence length, which is the wall you hit at 1m tokens. csa attacks this in two stages: (1) compress the kv cache by a factor `m` along the sequence dimension, then (2) sparsely select the top-`k` of those compressed blocks per query.

### stage 1: token-level compression

let `H ∈ R^{n×d}` be the input hidden states. csa first projects two parallel streams of kv entries and two streams of compression scores:

`C^a = H · W^a_KV,   C^b = H · W^b_KV ∈ R^{n×c}`

`Z^a = H · W^a_Z,    Z^b = H · W^b_Z ∈ R^{n×c}`

then it bundles every `m` tokens into one compressed entry. critically, the bundle for compressed slot `i` is built from **two adjacent windows** — the `m` tokens `[mi, m(i+1)-1]` from stream `a`, and the `m` tokens `[m(i-1), mi-1]` from stream `b`. learnable positional biases `B^a, B^b ∈ R^{m×c}` are added inside a softmax that normalizes across all `2m` elements jointly:

`[S^a_{mi:m(i+1)-1}; S^b_{m(i-1):mi-1}] = softmax_row( [Z^a_{mi:m(i+1)-1} + B^a; Z^b_{m(i-1):mi-1} + B^b] )`

`C^Comp_i = sum_{j=mi}^{m(i+1)-1} S^a_j ⊙ C^a_j  +  sum_{j=m(i-1)}^{mi-1} S^b_j ⊙ C^b_j`

a few subtleties:

- the softmax is row-wise across `2m` elements, so `S^a` and `S^b` *compete* for mass. each compressed entry is a learned convex combination of `2m` source tokens.
- the index ranges of `C^b` used in slot `i` overlap with those of `C^a` used in slot `i-1`. that is the "overlapped compression" trick — adjacent compressed entries share information about the boundary, so a query can't lose context by being just unlucky with block alignment.
- net effect: sequence length goes from `n` to `n/m`, but information mixing happens over a window of `2m`.

at slot `i = 0`, `Z^b` is padded with `-∞` and `C^b` with zeros so the softmax cleanly degenerates to a single-window compressor.

### stage 2: sparse selection via the lightning indexer

once we have `C^Comp ∈ R^{n/m × c}`, attending to all of it is still expensive at 1m tokens. csa runs a cheap **lightning indexer** to pick a top-`k` subset per query, inheriting the dsa idea from v3.2 but on a sequence that is already `m×` shorter.

the indexer reuses the same compression machinery to produce indexer keys `K^IComp ∈ R^{n/m × c_I}`, then for each query token `t`:

`c^Q_t = h_t · W_DQ ∈ R^{d_c}`

`q^I_t = c^Q_t · W_IUQ`  (split into `n^I_h` heads `q^I_{t,1}, ..., q^I_{t,n^I_h}`)

`w^I_t = h_t · W_w ∈ R^{n^I_h}`

`I_{t,s} = sum_{h=1}^{n^I_h} w^I_{t,h} · ReLU( q^I_{t,h} · K^IComp_s )`

two design choices to flag:

- the `ReLU` (rather than dot-product alone) makes the score nonnegative and pushes irrelevant blocks toward zero, which is friendlier to the top-`k` ranker.
- the entire indexer is run in **fp4**. it never produces the final attention output, only ranks blocks, so quantization noise mostly perturbs ties at the bottom of the top-`k` list.

then the top-`k` selector keeps:

`C^SprsComp_t = { C^Comp_s  :  I_{t,s} ∈ top-k(I_{t,:}) }`

### stage 3: core attention (mqa) over the selected blocks

core attention is multi-query: each compressed block serves as both key and value, and the latent query `c^Q_t` (shared with the indexer) is up-projected into `n_h` heads:

`q_t = c^Q_t · W_UQ ∈ R^{c·n_h}`

`o_{t,i} = CoreAttn(query = q_{t,i}, key = C^SprsComp_t, value = C^SprsComp_t)`

a few extra mechanics tucked in around the core:

- **rmsnorm** on each query head and on the (single) compressed kv head before the dot product, to keep logits from exploding.
- **partial rope** on the last 64 dims of queries and kv entries. since each kv entry is reused as both key *and* value, the naive output picks up an absolute position drift; csa applies rope at position `-i` to the i-th attention output to convert that drift back into a relative position signal.
- a **sliding-window branch** of `n_win` uncompressed kv entries, so a query can see fine-grained local detail (including tokens inside its own compressed block, which causal masking would otherwise hide).
- **attention sink**: a learnable per-head logit `z'_h` is added to the softmax denominator, `s_{h,i,j} = exp(z_{h,i,j}) / (sum_k exp(z_{h,i,k}) + exp(z'_h))`, letting heads abstain by sending mass into a non-token slot.

a **grouped output projection** then splits the `n_h` head outputs into `g` groups, projects each group down to `d_g`, then concatenates and projects to `d`. this avoids one giant `c·n_h → d` matmul.

### why csa is fast

at 1m tokens, csa effectively runs core attention on `min(k, n/m)` compressed blocks per query, with the indexer scoring `n/m` blocks in fp4. compared to dense attention over `n` tokens, the work per query drops by a factor of roughly `n / (k + n_win)` in the dense path and another factor of `m` is recovered in cache size. in v4 the kv entries themselves are stored in fp8 (with bf16 only for the rope-carrying dims), which pushes the kv cache down to ~10% of v3.2's at the same context length.

### a note on hca

csa's sibling, **heavily compressed attention (hca)**, drops sparse selection entirely. it runs the same kind of compressor but with `m' >> m` (and without the overlapped two-stream trick), getting `C^Comp ∈ R^{n/m' × c}`. then it just does dense mqa over those `n/m'` blocks. when `m'` is large enough (the report uses `m' = 128`), `n/m'` is small enough that dense attention is cheap. csa and hca are interleaved layer-by-layer — in v4-pro's 61-layer stack, layers 0–1 are hca, layers 2–60 alternate, and the final mtp block runs sliding-window only. the intuition: csa layers do *focused retrieval* over a long context; hca layers do *global low-resolution* mixing; alternating gives both at low cost.

## manifold-constrained hyper-connections (mhc)

the second piece is a residual-connection redesign. it sounds like a small thing — most papers don't bother explaining their residual stream — but at trillion-parameter scale, residual instability is one of the things that actually breaks training.

### standard hyper-connections, in one paragraph

vanilla residual: `x_{l+1} = x_l + F_l(x_l)`. **hyper-connections (hc)** generalize this by widening the residual stream from `R^d` to `R^{n_hc × d}`. for residual state `X_l ∈ R^{n_hc × d}` and learned linear maps `A_l ∈ R^{1×n_hc}`, `B_l ∈ R^{n_hc × n_hc}`, `C_l ∈ R^{n_hc × 1}`:

`X_{l+1} = B_l X_l + C_l F_l(A_l X_l)`

- `A_l X_l ∈ R^d` is the actual layer input (so `F_l` itself doesn't change shape).
- `B_l` mixes across the `n_hc` parallel residual streams.
- `C_l` projects the layer output back into the wider residual.

the upside is more expressive cross-layer routing. the downside, which deepseek hits at scale, is that nothing constrains `B_l` — its spectral norm can drift above 1, signal magnitudes can blow up across many stacked layers, and training becomes unstable.

### the manifold constraint

mhc fixes this by constraining `B_l` to live on the **birkhoff polytope** `M` — the set of `n × n` doubly stochastic matrices:

`M := { M ∈ R^{n×n}  |  M·1 = 1,  1^T·M = 1^T,  M ≥ 0 }`

three nice properties drop out immediately:

1. **non-expansiveness.** any doubly stochastic `M` has spectral norm `‖M‖_2 ≤ 1`. so `‖B_l X_l‖ ≤ ‖X_l‖`, both forward and backward. signal energy can never grow as it propagates through the residual mixer.
2. **closure under multiplication.** the product of doubly stochastic matrices is doubly stochastic. that means when you stack many layers, `prod_{i=1}^{L-l} B_{L-i}` is *still* in `M`, and the bound stays tight all the way through. you don't get "mostly bounded but the 80th layer ruins it."
3. **convex combination semantics.** `B_l X_l` is literally a convex combination of the rows of `X_l` — the row sums are 1 and entries are nonnegative. feature *means* are conserved exactly, and feature *norms* are bounded.

deepseek also constrains `A_l` and `C_l` to be nonnegative and bounded via sigmoids:

`A_l = σ(Ã_l)`,    `C_l = 2σ(C̃_l)`

so the input-mixing and output-injection mappings can't introduce sign flips that would catastrophically cancel signal.

### dynamic parameterization

the raw parameters are *input-dependent*. given `X_l`, mhc first flattens and normalizes:

`X̂_l = RMSNorm(vec(X_l)) ∈ R^{1 × n_hc·d}`

then it generates the unconstrained raw matrices as a learnable mix of dynamic and static parts:

`Ã_l = α_l^pre · (X̂_l W_l^pre) + S_l^pre`

`B̃_l = α_l^res · Mat(X̂_l W_l^res) + S_l^res`

`C̃_l = α_l^post · (X̂_l W_l^post)^T + S_l^post`

where `Mat(·)` reshapes a length-`n_hc^2` vector into an `n_hc × n_hc` matrix, the `S_l^*` are static learnable biases, and the gating scalars `α_l^*` are initialized small so the layer starts close to a static residual and gradually opens up the dynamic component.

### sinkhorn-knopp for the projection

projecting `B̃_l` onto `M` is the actual mathematical work. mhc uses the classic sinkhorn-knopp algorithm: exponentiate to enforce positivity, then iteratively normalize rows and columns:

`M^(0) = exp(B̃_l)`

`M^(t) = T_r( T_c( M^(t-1) ) )`

where `T_r` and `T_c` divide each row / column by its sum. this converges to a doubly stochastic matrix, and deepseek runs `t_max = 20` iterations, ending with `B_l = M^(t_max)`.

a couple of practical notes from the report:

- with `n_hc = 4`, `B_l` is only `4×4`, so the entire sinkhorn loop is essentially free per token — it's just bandwidth into the matmul, which deepseek fuses with custom kernels (the wall-clock overhead of the whole mhc machinery is reported at 6.7%).
- because `B_l` is small, the matmul is launch-bound; they use a split-k strategy to keep the gpu busy.
- selective recomputation and overlap with the dualpipe schedule absorb the activation-memory and pipeline-comm cost.

### why this matters for training

the practical claim of mhc is: you can keep the *expressivity* gains of hyper-connections (wider residual stream, dynamic mixing across streams) while preserving the *identity-mapping property* that made vanilla residuals trainable in the first place. when `B_l` is doubly stochastic, the layer composes like an averaging operator — energy-preserving and well-conditioned. when it isn't, you can technically train, but the loss landscape becomes hostile at depth.

deepseek's evidence here is mostly empirical (stable training at trillion-param scale, no spikes, comparable or better perplexity vs hc), but the underlying reason is structural: doubly stochastic mixers are exactly the right algebraic object to compose many times without blowing up.

## putting it together

at the system level, v4 is a moe (deepseekmoe ffns) with mhc residuals and an attention stack that alternates csa and hca, plus a sliding-window-only mtp head. the recurring theme across both csa and mhc is the same: **make the operator cheaper or more stable by constraining it to a structured subspace**, then pay the constant overhead of projection (softmax-gated convex combinations for csa's compressor, sinkhorn-knopp for mhc's residual mixer).

this is not a fundamentally new transformer. it's a careful set of architectural compromises that turn 1m-token context from "technically possible" into "actually cheap to run."
