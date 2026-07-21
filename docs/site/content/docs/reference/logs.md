---
title: "Logs"
weight: 3
---

# Logs

Everything both scripts produce lands in `logs/`: a JSON config per training run, a CSV
of per-epoch metrics, a JSON array per test run, and a small index of model ids.

## Training log — `model_pXXX-tXXX_<model_id>.json`

A single JSON object recording the run's configuration. This file is **required at
test time**: `test.py` reads `hd1`, `hd2` and `nrd` from it to rebuild the network.

```json
{
  "epochs": 128, "batch_size": 10000, "lr": 0.001, "dropout": 0.0,
  "K_min": 1, "K_max": 50, "weights": "",
  "epochs_dqn": 0, "dqn_walkers": 0, "dqn_round": false,
  "device_id": 0, "group_id": 34, "target_id": 0,
  "hd1": 1024, "hd2": 256, "nrd": 1,
  "model_name": "p034-t000", "model_mode": "MLP2RB",
  "model_id": 1784627982, "num_parameters": 501249
}
```

## Per-epoch metrics — `train_pXXX-tXXX_<model_id>.csv`

Written by `Trainer` alongside the JSON config, one row per epoch of the warm-up
phase. The Modified DQN phase writes its own `dqn_pXXX-tXXX_<model_id>.csv` in the same
format. Columns: `epoch`, `train_loss`, `vertices_seen`, `data_gen_time`,
`train_epoch_time`. Nothing reads these back — they are for your own plots.

## `model_id.txt`

One `model_id` per line, appended as models are trained. This is how
`read-test-logs-multiagent.py` finds the last `A` agents to aggregate.

## Test log — `test_pXXX-tXXX-<dataset>_<model_id>_<epoch>_B<B>.json`

A JSON array, one object per solved scramble:

```json
{
  "test_num": 0,
  "solution_length": 26,
  "attempts": 1,
  "time": 0.43,
  "moves": [2, 1, 3, 1, 1, 2, 1, 3, 0, 3, 0, 2, 1, 2, 0, 0, 2, 1, 3, 0, 0, 0, 0, 2, 0, 2],
  "vertex_num": "[9.64e+03, 4.27e+03, 2.57e+03]"
}
```

| Field | Meaning |
|---|---|
| `test_num` | Row index of the scramble in the dataset |
| `solution_length` | Number of moves found; `null` if unsolved |
| `attempts` | Which restart succeeded (see `--num_attempts`) |
| `time` | Wall-clock seconds for this scramble |
| `moves` | **Generator indices** applied in order to reach the solved state — index into the `actions` array of `generators/pXXX.json`; use `names` to render them |
| `vertex_num` | Node counts from the searcher's internal counter, as a string |

With `--return_tree 1`, the full search tree and starting state are additionally saved
as tensors under `forest/`.

`--shift` and `--skip_list` append `_shift<N>` and `_skip<list>` to the filename, so
runs over different slices of the same dataset do not overwrite each other. Note that
the aggregation scripts below match the plain form only.

## Aggregation scripts

### `read-test-logs.py`

```bash
python read-test-logs.py
```

Sweeps `logs/` for `santa` and `rnd` runs at epochs 16 and 128, takes the **minimum
solution length per scramble** across matching runs, and prints per-group test count,
solved percentage and average solution length — joined against the Santa 2023 best
scores for comparison. The datasets and epochs are hard-coded at the top; edit them for
other runs.

> **The script does not run as shipped.** It unconditionally reads
> `notebooks/avg-santa-scores.csv` for the Santa comparison column, and that file is
> not in the repository — so it fails with `FileNotFoundError`. Either supply the CSV
> (indexed by `group_target_id`, with `Santa Best` and `puzzle_type` columns) or delete
> the join block.

### `read-test-logs-multiagent.py`

```bash
python read-test-logs-multiagent.py [A] [EPOCH] [B]
```

Aggregates an ensemble on group `054` / `deepcubea`. Positional arguments (all
optional):

| Argument | Default | Meaning |
|---|---|---|
| `A` | `2` | How many of the most recent `model_id`s to analyse |
| `EPOCH` | `128` | Epoch in the log filenames |
| `B` | `1048576` | Beam width in the log filenames |

It reads the last `A` ids from `model_id.txt`, loads the matching test logs (warning
and skipping any that are missing), and prints per-agent statistics, ensemble
statistics from the shortest solution per scramble, and the winning agent's moves. See
[Multi-agent evaluation]({{< relref "/docs/usage/multiagent" >}}) for example output.
