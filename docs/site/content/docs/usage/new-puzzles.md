---
title: "Adding new puzzles"
weight: 4
math: true
---

# Adding new puzzles

Nothing in the model or the search is puzzle-aware, so adding a puzzle means adding
**data**, not code. Three files and a free `group_id`.

## 1. Generators — `generators/pXXX.json`

Define the legal moves as permutations, where `XXX` is your zero-padded `group_id`
(pick one not in the [group table]({{< relref "/docs/reference/groups" >}})):

```json
{
  "actions": [[...], [...]],
  "names":   ["A", "A'"]
}
```

Each entry of `actions` is a permutation of length `state_size`; all of them must have
the same length. `names` must pair every move `X` with its inverse `X'` — the
random-walk generator uses that pairing to avoid immediately backtracking, and the
generating set must be closed under inversion for the graph to be undirected.

## 2. Target — `targets/*.pt`

A 1-D `torch` tensor holding the solved state \( V_0 \): the class label of each
position. Its length is `state_size` and its distinct values determine `num_classes`,
which sets the model's one-hot input width.

```python
import torch
torch.save(torch.tensor([0, 0, 1, 1, 2, 2, ...]), "targets/pXXX-t000.pt")
```

## 3. Scrambles — `datasets/pXXX-tXXX-rnd.pt`

A 2-D tensor, one scramble per row, each row a complete state vector (states, not move
sequences). The simplest set is random walks far from the target:

```python
# apply ~10_000 random moves from V0, repeat 100 times, stack the resulting states
```

That is exactly how the shipped `rnd` datasets were built.

Full conventions — naming, shapes, dtypes — in
[Architecture → Data contract]({{< relref "/docs/architecture/data-contract" >}}).

## 4. Train and test

Pick `K_max` above the graph's diameter (if unknown, estimate it — random walks
should reliably reach "far" states):

```bash
python train.py --group_id XXX --target_id 0 --epochs 64 \
  --hd1 1024 --hd2 256 --nrd 1 --batch_size 10000 --K_max <diameter+margin> --device_id 0

python test.py --group_id XXX --target_id 0 --tests_num 10 --dataset rnd \
  --num_steps <2 × expected length> --epoch 64 --model_id {MODEL_ID} --B 65536 --verbose 1
```

## If it does not solve

| Symptom | Likely cause |
|---|---|
| Loss plateaus high, search wanders | `K_max` far below the diameter — deep states never appear in training |
| Solves easy scrambles, fails hard ones | Beam `--B` too small, or `--num_steps` below the true solution length |
| Search never terminates on solved states | Target tensor does not match the state encoding your generators produce |
| Random walks look degenerate | `names` inverse pairing is wrong, so walks backtrack constantly |

A useful sanity check before training at all: apply a handful of moves to \( V_0 \)
and confirm that applying the corresponding inverses returns exactly \( V_0 \).

## A caveat on what qualifies

The method assumes a **Cayley graph** — vertex-transitive, every state structurally
alike. The classical 15-puzzle (without wrap-around) violates this: the blank's
position changes the vertex degree. It is included in the project only as a non-Cayley
baseline, on the separate `puzzle-15` branch. Puzzles that break vertex-transitivity
may still work, but the theoretical footing is gone.
