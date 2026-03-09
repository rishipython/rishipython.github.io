---
layout: post
title: "test-time optimization in practice: alphaevolve and ttt-discover"
date: 2026-03-09
tags: [test-time-training, reinforcement-learning, alphaevolve, ttt-discover]
---

in many research settings, inference is no longer just "sample and rank." it is becoming an explicit optimization phase. two strong examples are **alphaevolve** and **ttt-discover**. both target the same meta-goal (find a new state-of-the-art solution), but they optimize different objects:

- alphaevolve optimizes the **candidate program** (external search over code).
- ttt-discover optimizes the **policy weights** at test time (internal search in parameter space).

this distinction matters, because it changes the objective, the exploration strategy, and the compute profile.

## alphaevolve: evolutionary search over code

references: [alphaevolve paper](https://arxiv.org/abs/2506.13131), [deepmind overview](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms){:target="_blank"}

alphaevolve can be viewed as a population-based optimization loop over programs.

let `p` denote a candidate program and `f(p)` its evaluator score (e.g., runtime, correctness, objective value). the core loop is:

1. sample one or more parent programs from an archive `A_t` (favoring high score, while preserving diversity).
2. use an llm to propose edits/mutations to obtain `p'`.
3. run verifiers/evaluators to compute `f(p')`.
4. insert `p'` into the archive if valid and competitive.
5. repeat.

in compact form, it is an evolutionary process:

`A_{t+1} = update(A_t, mutate(select(A_t)), f)`

where `mutate(.)` is llm-driven code editing and `f` is automated evaluation.

### why this works technically

alphaevolve separates "creative proposal" from "objective truth":

- the llm proposes candidate edits (high-variance, creative).
- the evaluator enforces correctness/performance (low-variance, objective).

that split is exactly what makes it practical for scientific optimization problems with executable verification.

### matrix multiplication result in algebraic terms

the matrix multiplication discovery can be described using bilinear algorithms. for multiplying `A` and `B`, a rank-`r` bilinear algorithm computes:

1. `m_t = (u_t^T vec(A)) * (v_t^T vec(B))`, for `t = 1..r`
2. `vec(C) = sum_{t=1}^r w_t * m_t`

each `m_t` is one scalar multiplication. so minimizing `r` minimizes scalar multiplications. alphaevolve found a `4x4` complex multiplication construction with `r = 48`, improving the longstanding previous construction in that setting.

intuitively, alphaevolve is searching over factorizations of the matrix multiplication tensor that preserve correctness while reducing multiplicative complexity.

## ttt-discover: reinforcement learning at test time

references: [learning to discover at test time](https://arxiv.org/abs/2601.16175), [project page](https://test-time-training.github.io/discover/){:target="_blank"}

ttt-discover frames one test problem as an rl environment and performs online training during inference.

define:

- `d`: problem description
- `s`: candidate solution state (e.g., code)
- `R(s)`: continuous reward from verifier/evaluator
- `pi_theta(a | d, s, c)`: policy that proposes an edit/action `a`
- transition `T(s, a) -> s'`

for each rollout:

1. choose an initial state `s_i` from a replay/archive buffer `H_i`
2. sample action `a_i ~ pi_{theta_i}( . | d, s_i, c_i )`
3. apply transition to get `s'_i = T(s_i, a_i)`
4. score with `r_i = R(s'_i)`
5. update both buffer and model weights

### reuse policy: puct-style state selection

ttt-discover does not restart from scratch every time. it reuses promising states using a puct-like score:

`score(s) = Q(s) + c * P(s) * sqrt(1 + T) / (1 + n(s))`

where:

- `Q(s)` tracks high downstream reward from expanding `s` (max-oriented in discovery),
- `P(s)` is a prior favoring better-ranked states,
- `n(s)` is visit count,
- `T` is total expansions,
- `c` controls exploration.

this is a key design choice: it extends effective horizon and balances exploitation/exploration over candidate solution trajectories.

### objective: risk-seeking toward top outcomes

in discovery, average reward is often the wrong objective. what matters is the best single solution found. ttt-discover therefore uses a risk-seeking/entropic objective that upweights high-reward rollouts (instead of optimizing plain expected reward).

practically, it also regularizes updates with a kl term to avoid destructive drift from the base policy, and adapts the objective sharpness during training.

conceptually:

- naive rl: "improve mean performance"
- discovery rl: "increase probability mass on extreme high-reward tails"

that difference is why it can outperform best-of-`N` under the same sampling budget on several tasks.

## side-by-side: alphaevolve vs ttt-discover

both methods are test-time optimization, but they allocate compute differently:

- **alphaevolve:** more compute in evaluator-driven program evolution; model weights mostly fixed.
- **ttt-discover:** more compute in online policy updates; state reuse plus rl fine-tuning on one problem instance.

alphaevolve is often easier to deploy when you already have robust external evaluators and want code-level auditability. ttt-discover is powerful when gradient-based adaptation can quickly specialize a strong base model to one hard instance.

## bottom line

the important shift is not just "more tokens at inference." it is **optimization at inference** with explicit objectives, replay/reuse, and verifier-guided feedback loops. that is a materially different regime from static prompting, and it is where many of the current state-of-the-art jumps are coming from.
