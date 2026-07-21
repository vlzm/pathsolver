---
title: "Pathsolver"
type: docs
---

# Pathsolver

**Zero-Knowledge ML for Finding Short Paths on Large Cayley Graphs.**
Pathsolver is the official implementation of the NeurIPS 2025 Spotlight paper
*"A Machine Learning Approach That Beats Large Rubik's Cubes"*
([arXiv:2502.13266](https://www.arxiv.org/pdf/2502.13266)). It finds short paths on
very large [Cayley graphs]({{< relref "/docs/research/problem-formulation" >}}) — the
state graphs of permutation puzzles such as the Rubik's Cube — by training a network
to estimate the *diffusion distance* to the solved state and using that estimate to
guide a [batched GPU beam search]({{< relref "/docs/research/method" >}}).

The approach uses **zero human knowledge**: no handcrafted heuristics, no domain
rules, no pattern databases. The same architecture and the same hyperparameters
transfer across puzzles unchanged.

<p align="center">
  <img src="images/fig.png" alt="Diffusion-distance predictor + batched beam search" style="max-width:100%;">
</p>

The whole pipeline — data generation, training, search — is plain PyTorch, built for
massive GPU parallelism. On a single NVIDIA H100, with model parameters
N₁ = 1024, N₂ = 256, Nᵣ = 1, beam width 2²⁰ and only 1.28×10⁸ training examples:

| Puzzle | Graph size | Training (min) | Solving (min) | Avg. solution len. | Solved (%) |
|:---|:---:|---:|---:|---:|---:|
| Rubik's Cube 3×3×3 | 4×10¹⁹ | 0.8 | 0.17 | 20.5 ± 0.1 | 100 |
| Rubik's Cube 4×4×4 | 7×10⁴⁵ | 1.5 | 1.24 | 65.0 ± 1.0 | 94 |
| Pancake Graph 55 | 3×10⁷³ | 1.4 | 3.49 | 50.0 ± 0.3 | 100 |
| Klotski 6×6 (PBC) | 2×10⁴¹ | 1.2 | 0.45 | 113 ± 2.0 | 100 |

With more compute the method reaches **98% optimality** on 3×3×3 (QTM), outperforms
the top Santa Kaggle 2023 submissions on 3×3×3, 4×4×4 and 5×5×5 in average solution
length, and on 3×3×3 delivers **20× faster** path finding than the previous state of
the art at comparable solution lengths. See [Results]({{< relref "/docs/research/results" >}}).

## Quick links

- 📄 **Paper** — [arXiv:2502.13266](https://www.arxiv.org/pdf/2502.13266) (NeurIPS 2025 Spotlight)
- 💻 **Code** — [github.com/vlzm/pathsolver](https://github.com/vlzm/pathsolver)
- 🧩 **Browser demo** — client-side 3×3×3 solver in [`ui/web/cube/`]({{< relref "/docs/usage/onnx-and-ui" >}})

## Where to go next

| I want to… | Start here |
|---|---|
| Understand what this project does | [Introduction]({{< relref "/docs/introduction" >}}) |
| Solve a puzzle with pretrained weights | [Testing a pretrained model]({{< relref "/docs/usage/testing" >}}) |
| Train my own model | [Training]({{< relref "/docs/usage/training" >}}) |
| Add my own puzzle | [Adding new puzzles]({{< relref "/docs/usage/new-puzzles" >}}) |
| See how the code fits together | [Architecture]({{< relref "/docs/architecture" >}}) |
| Read the method and the numbers | [Research]({{< relref "/docs/research" >}}) |
| Look up a flag, a group ID or a term | [Reference]({{< relref "/docs/reference" >}}) |
