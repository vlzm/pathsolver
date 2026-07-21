---
title: "CLI"
weight: 1
---

# CLI reference

Complete flag lists, taken from the `argparse` definitions in `train.py` and
`test.py`. (The README shows only the commonly used subset.)

## `train.py`

```bash
python train.py --group_id <id> [options]
```

### Puzzle

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--group_id` | int | **required** | Group ID — which puzzle. See [Groups]({{< relref "/docs/reference/groups" >}}) |
| `--target_id` | int | `0` | Target ID — which solved state |

### Model architecture

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--hd1` | int | `1024` | Size of the first hidden layer |
| `--hd2` | int | `256` | Size of the second hidden layer (`0` disables it) |
| `--nrd` | int | `4` | Number of residual blocks (`0` disables them) |
| `--dropout` | float | `0.0` | Dropout rate |

### Optimisation

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--epochs` | int | `256` | Number of training epochs |
| `--batch_size` | int | `10000` | Batch size |
| `--lr` | float | `0.001` | Learning rate |
| `--K_min` | int | `1` | Minimum K for random walks |
| `--K_max` | int | `30` | Maximum K for random walks |
| `--weights` | str | `""` | Warm-start from this checkpoint name (no extension) in `weights/` |

### Modified DQN phase

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--epochs_dqn` | int | `0` | Number of Modified DQN epochs (`0` disables the phase) |
| `--dqn_walkers` | int | `0` | Walkers per DQN epoch (`0` = 1/10 of the warm-up walkers) |
| `--dqn_round` | flag | off | Round DQN targets to integers |

### Runtime

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--device_id` | int | `0` | CUDA device index (ignored on CPU) |

> Note the defaults differ from the published configuration: the paper uses
> `--hd2 256 --nrd 1`, so pass `--nrd 1` explicitly to reproduce it.

## `test.py`

```bash
python test.py --group_id <id> --model_id <id> --epoch <n> [options]
```

### What to load

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--group_id` | int | **required** | Group ID |
| `--target_id` | int | `0` | Target ID |
| `--model_id` | int | **required** | Model ID (`int(time.time())` from training; `333`/`444`/`555` for released weights) |
| `--epoch` | int | **required** | Which epoch's checkpoint to load |
| `--dataset` | str | `rnd` | Scramble set: `rnd`, `santa`, `deepcubea`, `deepcubeadifficult`, `deepcubeahard` |

### Search

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--B` | int | `2**18` | Beam width |
| `--num_steps` | int | `200` | Max steps per beam search run |
| `--num_attempts` | int | `2` | Number of restarts |

### Which scrambles

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--tests_num` | int | `10` | Number of tests to run |
| `--shift` | int | `0` | Skip this many scrambles from the start of the dataset |
| `--skip_list` | str | — | IDs to skip, e.g. `'[2, 5]'` or `'2,5'` |

### Runtime and output

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--device_id` | int | `0` | CUDA device index |
| `--verbose` | int | `0` | Show a `tqdm` progress bar if > 0 |
| `--return_tree` | int | `0` | Save the beam search tree to the `forest/` folder |

`test.py` takes no architecture flags: it reads `hd1`, `hd2` and `nrd` back from the
training log `logs/model_pXXX-tXXX_<model_id>.json` and rebuilds the network from
those before loading
`weights/pXXX-tXXX_<model_id>_e<epoch:05d>.pth`. Both files must be present for a
`model_id` to be usable.
