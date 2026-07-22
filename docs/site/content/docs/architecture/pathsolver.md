---
title: "The pathsolver package"
weight: 2
math: true
---

# The `pathsolver` package

About a thousand lines total. Five modules, no inheritance hierarchies, no config
framework — each file does one thing.

| Module | Lines | Responsibility |
|---|---:|---|
| `model.py` | 116 | The `PathSolver` network and batched inference helpers |
| `trainer.py` | 184 | Random-walk generation, training loop, checkpointing |
| `searcher.py` | 143 | Batched GPU beam search |
| `dqn.py` | 44 | Bellman (Modified DQN) target refinement |
| `utils.py` | 54 | Puzzle loading, inverse moves, neighbours, hashing |

## `model.py` — the network

`PathSolver(state_size, hd1=5000, hd2=1000, nrd=2, output_dim=1, dropout_rate=0.0, num_classes=6)`
is a plain MLP with residual blocks:

```text
state [N, state_size]
  → one-hot [N, state_size * num_classes]
  → Linear(hd1) → BatchNorm → ReLU → Dropout
  → Linear(hd2) → BatchNorm → ReLU → Dropout    (skipped if hd2 == 0)
  → nrd × ResidualBlock(hd2)                    (skipped if nrd == 0 or hd2 == 0)
  → Linear(1)                                   → scalar distance estimate
```

A `ResidualBlock` is `Linear → BN → ReLU → Dropout → Linear → BN → (+ residual) → ReLU`
at constant width. `batch_process` runs the network over arbitrarily large tensors in
fixed-size chunks, which is what the searcher uses to score a full beam.

`num_classes` is the number of distinct labels in a state (6 for cube faces); it only
affects the one-hot width.

## `trainer.py` — data generation and the loop

`Trainer` owns everything about training:

- `do_random_step(states, last_moves)` — one random move per row, masking the inverse
  of the previous move so walks don't immediately backtrack.
- `generate_random_walks(k, K_min, K_max)` — launches `k` walkers per length, returns
  `(states, Y)` shuffled, where `Y` is the walk length used as the regression label.
  `walkers_num` defaults to `1_000_000 // K_max`, keeping epoch size ~10⁶ states.
- `_train_epoch(X, Y)` — MSE + Adam over `batch_size` chunks.
- `run(...)` / `run_dqn(...)` — the epoch loops; `run_dqn` replaces the random-walk
  labels with `bellman_targets` before optimising.

The instance sets `self.id = int(time.time())` — this is the **`model_id`** you pass
to `test.py`. Checkpoints go to `weights/`, per-epoch metrics to `logs/`.

## `searcher.py` — batched beam search

`Searcher(model, all_moves, V0, device, verbose)` holds a random `hash_vec` used to
fingerprint states. The entry point is
`get_solution(state, B, num_steps, num_attempts, return_tree)`; the core loop
(`do_greedy_step`):

1. `get_neighbors(states)` — expand all `n_gens` neighbours of every beam state with a
   single `torch.gather` on an expanded permutation table.
2. `state2hash` + `get_unique_hashed_states_idx` — hash, drop anything already visited
   (`states_bad_hashed`), then sort and drop duplicates within the batch.
3. Score the survivors through `batch_process`.
4. `torch.argsort(value)[:B]` — keep the `B` lowest-scoring states as the next beam.

Everything is chunked by `self.batch_size = 2**14` so GPU memory stays bounded
independently of `B`. `self.counter` tracks expanded/deduplicated node counts for
reporting.

> Hashing is a dot product with a random int64 vector (entries drawn below 10¹⁵) —
> collisions are possible in principle. `--num_attempts` restarts with a fresh seed, which also re-rolls the
> hashes.

## `dqn.py` — Bellman refinement

One function, `bellman_targets(net, X, y_rw, all_moves, V0, ...)`:

```katex
y(g) = \mathrm{clamp}\Big(\min\big(1 + \min_{s \in S} V_\theta(gs),\; y_{\mathrm{rw}}(g)\big),\; \ge 1\Big),
\quad y(V_0) = 0
```

It expands neighbours per chunk sized so the model always sees ~`batch_size` rows.
`flag_round` rounds targets to integers — reported in the paper as no improvement, and
off by default.

## `utils.py` — the primitives

- `generate_inverse_moves(moves)` — pairs `X` with `X'` by name to build the inverse
  index used by random walks.
- `get_neighbors(states, all_moves, batch_size)` — `[N, state_size]` →
  `[N, n_gens, state_size]`, chunked.
- `state2hash(states, hash_vec, batch_size)` — `[N, state_size]` → `[N]` int64.

These three are the puzzle-agnostic substrate `trainer.py` and `searcher.py` are built
from.

> `utils.py` also defines `load_cube_data(cube_size, cube_type, device)`, which reads
> `generators/{cube_type}_cube{cube_size}.json`. No file in `generators/` uses that
> naming scheme, and although `train.py` imports the function it never calls it — both
> `train.py` and `test.py` open `generators/pXXX.json` inline. Treat it as dead code.
