---
title: "Groups"
weight: 2
---

# Available groups and Kmax

Every puzzle has a `group_id`, passed as `--group_id` and written `pXXX` in filenames.
A group is usable when both `generators/pXXX.json` and `targets/pXXX-t000.pt` exist.

`Kmax` is a suggested random-walk length — pass it as `--K_max`
([Training]({{< relref "/docs/usage/training" >}})). Treat it as guidance, not as the
literal published setting: the reproduction scripts pass slightly different values for
some groups (see the note below).

| Group ID | Puzzle | Kmax |
|---|---|---|
| 000 | Cube 2x2x2 | 15 |
| 001 | Cube 3x3x3 | 26 |
| 002 | Cube 4x4x4 | 45 |
| 003 | Cube 5x5x5 | 65 |
| 004 | Cube 6x6x6 | 150 |
| 005 | Cube 7x7x7 | — |
| 006 | Cube 8x8x8 | — |
| 007 | Cube 9x9x9 | — |
| 008 | Cube 10x10x10 | — |
| 009 | Cube 19x19x19 | — |
| 010 | Cube 33x33x33 | — |
| 011 | Wreath 6/6 | 10 |
| 012 | Wreath 7/7 | 10 |
| 013 | Wreath 12/12 | 20 |
| 014 | Wreath 21/21 | 35 |
| 015 | Wreath 33/33 | 75 |
| 017 | Globe 1/8 | 60 |
| 018 | Globe 1/16 | 110 |
| 019 | Globe 2/6 | 25 |
| 020 | Globe 3/4 | 40 |
| 021 | Globe 6/4 | 40 |
| 022 | Globe 6/8 | 165 |
| 023 | Globe 6/10 | 170 |
| 024 | Globe 3/33 | 500 |
| 025 | Globe 8/25 | 700 |
| 026 | Puzzle 8 | 30 |
| 027 | Puzzle 15 | 80 |
| 028 | Puzzle 24 | 150 |
| 029 | Puzzle 35 | 200 |
| 030 | Puzzle 48 | 250 |
| 031 | Puzzle 63 | 300 |
| 032 | Puzzle 80 | — |
| 033 | Puzzle 99 | — |
| 034 | LRX 10 | 50 |
| 035 | LRX 15 | 100 |
| 036 | LRX 20 | 200 |
| 037 | LRX 25 | 300 |
| 038 | LRX 30 | 300 |
| 039 | LRX 35 | — |
| 040 | LRX 40 | — |
| 041 | LRX 45 | — |
| 042 | LRX 50 | — |
| 043 | LRX 55 | — |
| 044 | Pancake 10 | 15 |
| 045 | Pancake 15 | 25 |
| 046 | Pancake 20 | 30 |
| 047 | Pancake 25 | 40 |
| 048 | Pancake 30 | 45 |
| 049 | Pancake 35 | 55 |
| 050 | Pancake 40 | 60 |
| 051 | Pancake 45 | 65 |
| 052 | Pancake 50 | 70 |
| 053 | Pancake 55 | 75 |
| 054 | Cube 3x3x3 (DeepCubeA metric) | 26 |

## Notes

- **`—` in the Kmax column.** The group ships generators and a target, but the README
  table gives no value and no reproduction script trains it. Pick `K_max` yourself —
  it only needs to comfortably exceed the graph diameter.
- **Kmax vs. the reproduction scripts.** Where the two disagree, the scripts are what
  produced the published numbers: group `001` uses 25 (not 26), `002` 60 (not 45),
  `003` 70 (not 65), `004` 100 for `t000` and 150 for `t001`, `017` 45 for `t000` and
  60 for `t001`, `020` 30 for `t000` and 40 for `t001`. See the `triples` job lists in
  `traintest-tab4-santa.sh` and `traintest-tab4-rnd.sh`.
- **The LRX groups (`034`–`043`).** LRX is the Cayley graph of the symmetric group
  with generators L (cyclic shift left), R (cyclic shift right) and X (transposition
  of the first two positions), studied in the *CayleyPy RL* paper
  ([arXiv:2502.18663](https://arxiv.org/abs/2502.18663)) — including support for the
  OEIS-A186783 conjecture that its diameter is n(n−1)/2. These groups pair naturally
  with the [Modified DQN phase]({{< relref "/docs/usage/training#bellman-modified-dqn-refinement" >}})
  from the same paper.
- **`generators/p016.json` has no target.** It exists (4 wreath-style generators, state
  size 198) but there is no `targets/p016-t000.pt`, so the group cannot be trained or
  tested as shipped — hence its absence from the table.
- **Group `001` vs `054`.** Both are the 3×3×3 cube; they differ in the move metric.
  `001` uses UQTM (the Santa 2023 move set), `054` uses the QTM metric used by
  DeepCubeA — which is what the optimality numbers in
  [Results]({{< relref "/docs/research/results" >}}) are measured against. The
  `deepcubea*` datasets exist only for `054`.
- **Reproducing Table 4.** Use `traintest-tab4-santa.sh` and `traintest-tab4-rnd.sh`.
  The exact generators and scrambles per experiment are in `paper/solver-scrambles`
  and `paper/figure-scrambles`.
- **Puzzle 15 caveat.** The classical 15-puzzle (no wrap-around) is *not* a Cayley
  graph — node degree varies with the blank's position, so the state graph is not
  vertex-transitive. It is kept as a non-Cayley baseline on the separate
  [`puzzle-15`](https://github.com/khoruzhii/cayleypy-cube/tree/puzzle-15) branch. The
  `Puzzle N` entries above use periodic boundary conditions and *are* Cayley graphs.

To add your own group, pick a free ID and follow
[Adding new puzzles]({{< relref "/docs/usage/new-puzzles" >}}).
