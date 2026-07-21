---
title: "Problem formulation"
weight: 1
math: true
---

# Problem formulation

## Cayley graphs

Let \( G \) be a finite group and \( S = \{s_1, \dots, s_n\} \subset G \) a generating
set closed under inversion. The **Cayley graph** \( \mathrm{Cay}(G, S) \) has the
elements of \( G \) as vertices, with an edge \( g \to g s_i \) for every
\( g \in G \), \( s_i \in S \).

Two properties matter here:

- The graph is **vertex-transitive** — every vertex looks the same. Finding a path
  between arbitrary \( g \) and \( h \) therefore reduces to finding a path from
  \( h^{-1} g \) to the identity \( e \), i.e. to *solving* a single scrambled state.
- The graph is **implicit**. It is defined by \( |S| \) permutations, but its vertex
  count is \( |G| \) — far beyond anything that can be stored.

For permutation puzzles, states are permutations of a coloured/labelled vector, and
generators are the legal moves. In the repository a puzzle is exactly this: a tensor
`all_moves` of shape `[n_gens, state_size]` (see
[Data contract]({{< relref "/docs/architecture/data-contract" >}})).

| Puzzle | \( \lvert G \rvert \) |
|---|---:|
| Rubik's Cube 3×3×3 | \( 4 \times 10^{19} \) |
| Rubik's Cube 4×4×4 | \( 7 \times 10^{45} \) |
| Klotski 6×6 (PBC) | \( 2 \times 10^{41} \) |
| Pancake graph 55 | \( 3 \times 10^{73} \) |

> The classical 15-puzzle (without wrap-around) is **not** a Cayley graph: vertex
> degree depends on where the blank is, so the graph is not vertex-transitive. It is
> included only as a non-Cayley baseline, on the separate
> [`puzzle-15`](https://github.com/khoruzhii/cayleypy-cube/tree/puzzle-15) branch.

## The task

Given a scrambled state \( g \), find a short word
\( s_{i_1} s_{i_2} \cdots s_{i_L} \) in the generators taking \( g \) to \( e \).
Two objectives are in tension:

1. **Solve rate** — reach \( e \) at all, within a step and memory budget.
2. **Solution length** \( L \) — how close to the true graph distance
   \( d(g, e) \), i.e. God's number for that state.

## Why classical search does not scale

- **BFS / bidirectional BFS** needs to materialise a frontier that grows like
  \( |S|^{d} \). For 3×3×3 in QTM, the frontier at depth 10 is already billions of
  states.
- **IDA\* with pattern databases** (the classic optimal Rubik's solvers) requires
  hand-designed, puzzle-specific abstractions and precomputed tables — human
  knowledge that does not transfer to the Pancake graph or Klotski.
- **DeepCubeA-style learned heuristics** remove the human knowledge but rely on
  value iteration over sampled states plus a weighted-A\* search whose node expansion
  is largely sequential — the bottleneck this work removes.

The alternative pursued here: learn a cheap, GPU-batchable scalar estimate of distance
and lean entirely on parallelism at search time. See
[Method]({{< relref "/docs/research/method" >}}).
