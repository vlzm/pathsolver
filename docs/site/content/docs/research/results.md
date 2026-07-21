---
title: "Results"
weight: 3
math: true
---

# Results

All numbers below are from the NeurIPS 2025 paper
([arXiv:2502.13266](https://www.arxiv.org/pdf/2502.13266)) and the reproduction
scripts in the repository (`traintest-tab4-santa.sh`, `traintest-tab4-rnd.sh`).

## One configuration, four graphs

Single NVIDIA H100, model parameters \( N_1 = 1024 \), \( N_2 = 256 \),
\( N_r = 1 \), beam width \( B = 2^{20} \), \( 1.28 \times 10^{8} \) training
examples. **The architecture and hyperparameters are identical across rows** — only
the generator set and \( K_{\max} \) change.

| Puzzle | Graph size | Training (min) | Solving (min) | Avg. solution len. | Solved (%) |
|:---|:---:|---:|---:|---:|---:|
| Rubik's Cube 3×3×3 | \( 4 \times 10^{19} \) | 0.8 | 0.17 | 20.5 ± 0.1 | 100 |
| Rubik's Cube 4×4×4 | \( 7 \times 10^{45} \) | 1.5 | 1.24 | 65.0 ± 1.0 | 94 |
| Pancake Graph 55 | \( 3 \times 10^{73} \) | 1.4 | 3.49 | 50.0 ± 0.3 | 100 |
| Klotski 6×6 (PBC) | \( 2 \times 10^{41} \) | 1.2 | 0.45 | 113 ± 2.0 | 100 |

Training in under two minutes is the headline: the cost of the method is dominated by
search, not by learning.

## Against prior work

- **Optimality on 3×3×3 (QTM).** With a larger compute budget the method reaches
  **98% optimality** on the QTM metric — i.e. 98% of solutions match the known optimal
  length. Benchmarked on the DeepCubeA scramble sets, available in the repository as
  the `deepcubea` (1000 scrambles), `deepcubeadifficult` (16) and `deepcubeahard` (69)
  datasets under group `054`.
- **Speed.** On 3×3×3, at comparable solution lengths, the batched GPU beam search
  finds paths **20× faster** than the previous state of the art (DeepCubeA /
  EfficientCube), whose node expansion is far less parallel-friendly.
- **Santa 2023 (Kaggle).** The method **outperforms the top Santa 2023 submissions**
  on 3×3×3, 4×4×4 and 5×5×5 in average solution length — despite those submissions
  being heavily puzzle-specific.

Optimal lengths for the 16-scramble subset (shipped as `deepcubeadifficult`), for
reference when checking your own runs:

```text
[20, 20, 20, 21, 20, 20, 20, 20, 19, 20, 20, 19, 21, 20, 20, 20]
```

## Reproducing

| Goal | How |
|---|---|
| Solve with released weights | [Usage → Testing]({{< relref "/docs/usage/testing" >}}) |
| Table 4 (Santa / random scrambles) | `traintest-tab4-santa.sh`, `traintest-tab4-rnd.sh` |
| Ensemble / multi-agent numbers | [Usage → Multi-agent evaluation]({{< relref "/docs/usage/multiagent" >}}) |
| Exact generators & scrambles per experiment | `paper/solver-scrambles`, `paper/figure-scrambles` |

Results land in `logs/` as JSON; see
[Reference → Logs]({{< relref "/docs/reference/logs" >}}) for the format and the
aggregation scripts.
