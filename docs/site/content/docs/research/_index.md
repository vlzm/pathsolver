---
title: "Research"
weight: 3
bookCollapseSection: true
---

# Research

The scientific content behind Pathsolver, following two papers in the CayleyPy
series:

- **[arXiv:2502.13266](https://www.arxiv.org/pdf/2502.13266)** — *"A Machine Learning
  Approach That Beats Large Rubik's Cubes"* (NeurIPS 2025 Spotlight). The core method:
  diffusion-distance regression + batched GPU beam search, and all headline results.
- **[arXiv:2502.18663](https://arxiv.org/abs/2502.18663)** — *"CayleyPy RL:
  Pathfinding and Reinforcement Learning on Cayley Graphs"*. Combines the
  diffusion-distance approach with reinforcement learning: the
  [Modified DQN refinement]({{< relref "/docs/research/method#bellman-refinement-modified-dqn" >}})
  implemented in `pilgrim/dqn.py`, and the study of the LRX Cayley graph
  (cyclic shift + transposition), including support for the OEIS-A186783 conjecture
  that its diameter is n(n−1)/2.

- **[Problem formulation]({{< relref "/docs/research/problem-formulation" >}})** —
  Cayley graphs, permutation puzzles, and why exact search does not scale.
- **[Method]({{< relref "/docs/research/method" >}})** — learning the diffusion
  distance and using it to steer a batched GPU beam search.
- **[Results]({{< relref "/docs/research/results" >}})** — benchmarks against
  DeepCubeA, EfficientCube and the Santa 2023 Kaggle solutions.
