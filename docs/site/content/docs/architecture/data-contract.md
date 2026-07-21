---
title: "Data contract"
weight: 3
math: true
---

# Data contract

A puzzle is defined entirely by files. Nothing about it lives in code, so if your
files satisfy the conventions below, `train.py` and `test.py` work unchanged. This is
the contract to preserve when [adding a new puzzle]({{< relref "/docs/usage/new-puzzles" >}}).

## Identifiers

Two integers identify every artefact:

- **`group_id`** — the puzzle (its generating set). Formatted as three digits `pXXX`
  in filenames: `--group_id 54` → `p054`. The full table is in
  [Reference → Groups]({{< relref "/docs/reference/groups" >}}).
- **`target_id`** — which solved state to aim for, formatted `tXXX`. Almost always
  `0`; it exists because a graph can have several meaningful targets.

## `generators/pXXX.json` — the moves

```json
{
  "actions": [[0, 3, 1, 2, ...], [2, 0, 3, 1, ...]],
  "names":   ["R", "R'"]
}
```

- `actions` — a list of `n_gens` permutations, each of length `state_size`. Loaded as
  an int64 tensor `all_moves` of shape `[n_gens, state_size]`. A move is applied with
  `torch.gather(state, 1, all_moves[i])`, so entry `j` of a permutation is *the index
  the value at position `j` comes from*.
- `names` — one label per action. These are not decoration: `generate_inverse_moves`
  pairs `"X"` with `"X'"` by string matching to build the inverse-move index used
  during random-walk generation. **Every move must have its inverse present in the
  set**, named with a trailing apostrophe. There is no self-inverse special case: a
  move whose name has no `'`-suffixed partner makes `generate_inverse_moves` raise
  `ValueError`, so list a self-inverse move twice, once under each name.

The generating set must be closed under inversion — otherwise the graph is directed
and the whole formulation breaks.

## `targets/*.pt` — the solved state

A 1-D `torch` tensor \( V_0 \) of length `state_size`, holding the label of each
position in the solved configuration. Values are class indices in
`[0, num_classes)` — for a 3×3×3 cube, the six face colours.

`num_classes` follows from this tensor and sets the one-hot input width of the model.

## `datasets/pXXX-tXXX-{rnd,santa}.pt` — scrambles

A 2-D tensor, one scramble per row, each row a full state vector of length
`state_size` (states, not move sequences). The suffix names the source:

| Suffix | Content |
|---|---|
| `rnd` | 100 scrambles, generated with 10 000 (+1) random steps from \( V_0 \) (the generating script is not in the repository); present for most groups but **not** for `054` |
| `santa` | The official Kaggle Santa 2023 scrambles |
| `deepcubea` | 1000 DeepCubeA scrambles — group `054` only |
| `deepcubeadifficult` | 16-scramble DeepCubeA subset — group `054` only |
| `deepcubeahard` | 69-scramble DeepCubeA subset — group `054` only |

> **Mismatch with the README.** The README describes `deepcubeadifficult` as the
> 69-scramble subset and `deepcubeahard` as the 16-scramble one. The tensors shipped in
> `datasets/` are the other way round — `p054-t000-deepcubeadifficult.pt` is
> `(16, 54)`, `p054-t000-deepcubeahard.pt` is `(69, 54)`. The table above reflects the
> files. The published list of 16 optimal lengths therefore matches
> `deepcubeadifficult` as shipped.

`--dataset` selects the suffix; `--tests_num` caps how many rows from the top are
used, and `--shift` offsets the starting row.

## `weights/` — checkpoints

`weights/pXXX-tXXX_<model_id>_e<epoch:05d>.pth`, where `model_id = int(time.time())`
at training start. The released checkpoints use readable ids instead: `333`, `444`,
`555`.

A checkpoint is only loadable together with its **training log**,
`logs/model_pXXX-tXXX_<model_id>.json` — `test.py` reads `hd1`, `hd2` and `nrd` from
there to rebuild the network before loading the state dict. Keep the two files
together.

## `logs/` — results

`logs/test_pXXX-tXXX-{dataset}_{model_id}_{epoch}_B{B}.json`, one JSON array per run.
See [Reference → Logs]({{< relref "/docs/reference/logs" >}}).
