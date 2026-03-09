---
layout: post
title: "test-time training: alphaevolve and ttt-discover"
date: 2026-02-16
tags: [test-time-training, reinforcement-learning, alphaevolve, ttt-discover]
---

the paradigm is shifting from massive pre-training to massive test-time compute. two recent methods, **alphaevolve** and **ttt-discover**, demonstrate how we can turn inference into an optimization loop, allowing models to "think" and improve their own outputs on the fly.

## alphaevolve: evolutionary coding agents

[arxiv:2506.13131](https://arxiv.org/abs/2506.13131){:target="_blank"}

alphaevolve is an evolutionary coding agent that iteratively improves algorithms through direct code modifications. instead of just generating a solution once, it orchestrates an autonomous pipeline where llms propose changes, evaluate them against a verifier, and evolve the codebase based on feedback.

the results are non-trivial. it discovered a matrix multiplication algorithm for 4x4 complex matrices that uses only 48 scalar multiplications—the first improvement over strassen's algorithm in 56 years. it’s also being used to optimize data center scheduling and hardware accelerator designs.

## ttt-discover: rl at inference time

[project page](https://test-time-training.github.io/discover/){:target="_blank"}

while alphaevolve optimizes code, **ttt-discover** (test-time training to discover) optimizes the model itself. developed by researchers at stanford, nvidia, and together ai, it performs reinforcement learning *during inference*.

instead of the standard "best-of-n" sampling where you generate 100 solutions and pick the best one, ttt-discover actually updates the model's weights on the specific problem instance using continuous reward signals from a verifier. the model effectively specializes itself into a narrow expert for that single problem.

they've demonstrated wins across mathematics (improving bounds on the erdős minimum overlap problem), gpu kernel optimization (2x faster matrix multiplication kernels on h100), and even biology.

## the takeaway

we are moving away from the idea that a model's knowledge is frozen after pre-training. by allocating more compute to inference—whether through evolutionary code search or gradient-based weight updates—we can solve problems that are impossible for a fixed, generalist model. inference is no longer just a forward pass; it's a search process.
