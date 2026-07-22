---
title: "Training"
weight: 2
math: true
---

# Training

Training needs no dataset on disk — random walks are generated on the GPU inside the
loop. All you supply is a puzzle (`--group_id`), a target (`--target_id`) and a
budget.

```bash
python train.py --group_id <id> --target_id 0 --epochs <epochs> \
  --hd1 <N_1> --hd2 <N_2> --nrd <N_r> --batch_size 10000 --K_max <K_max> --device_id 0
```

Example — the 24-puzzle:

```bash
python train.py --group_id 28 --target_id 0 --epochs 16 \
  --hd1 1024 --hd2 512 --nrd 1 --batch_size 10000 --K_max 100 --device_id 0
```

## Hyperparameters

| Flag | Default | Meaning |
|---|---:|---|
| `--hd1` | 1024 | Width of the first hidden layer (\( N_1 \)) |
| `--hd2` | 256 | Width of the second hidden layer (\( N_2 \)); `0` disables it |
| `--nrd` | 4 | Number of residual blocks (\( N_r \)); `0` disables them |
| `--batch_size` | 10000 | SGD batch size |
| `--lr` | 0.001 | Adam learning rate |
| `--dropout` | 0.0 | Dropout rate |
| `--epochs` | 256 | Random-walk regression epochs |
| `--K_min` / `--K_max` | 1 / 30 | Random-walk length range |
| `--weights` | — | Checkpoint name in `weights/` to warm-start from |

The paper's headline results use \( N_1 = 1024 \), \( N_2 = 256 \), \( N_r = 1 \) for
*every* puzzle — so the architecture flags are usually not what you should be tuning.

**`--K_max` is the one genuinely per-puzzle knob.** It bounds the random-walk length,
so it must exceed the graph's diameter — too small and the model never sees deep
states; too large and epochs are wasted on states that are all equally far away. The
values used per puzzle are tabulated in
[Reference → Groups]({{< relref "/docs/reference/groups" >}}). Note that epoch size is
`1_000_000 // K_max` walkers per length, i.e. ~10⁶ states per epoch regardless of
`K_max`.

## Bellman (Modified DQN) refinement

After the random-walk phase you can refine the value estimates with a Bellman update
on the Cayley graph ([method]({{< relref "/docs/research/method" >}})). The phase
implements the Modified DQN of the *CayleyPy RL* paper
([arXiv:2502.18663](https://arxiv.org/abs/2502.18663)):

| Flag | Default | Meaning |
|---|---:|---|
| `--epochs_dqn` | 0 | Number of Modified DQN epochs; `0` disables the phase |
| `--dqn_walkers` | 0 | Walkers per DQN epoch; `0` means 1/10 of the warm-up walkers |
| `--dqn_round` | off | Round DQN targets to integers |

`--dqn_round` is reported in that paper as giving no improvement; it is there for
reproduction.

## Finding your model afterwards

Training assigns `model_id = int(time.time())` at start and prints it. Checkpoints
land in `weights/` and metrics in `logs/`. Pass that id back to `test.py`:

```bash
python test.py --group_id 28 --target_id 0 --tests_num 3 --dataset rnd \
  --num_steps 300 --num_attempts 1 --verbose 1 \
  --epoch 16 --model_id {MODEL_ID} --B 65536 --device_id 0
```

Replace `{MODEL_ID}` with the numeric identifier from the logs, and make sure
`--epoch` names a checkpoint that was actually written. You do **not** re-specify the
architecture: `test.py` reads `hd1`/`hd2`/`nrd` back from
`logs/model_pXXX-tXXX_<model_id>.json`, so keep that file next to the checkpoint.

Next: [Multi-agent evaluation]({{< relref "/docs/usage/multiagent" >}}) to train an
ensemble, or [Testing]({{< relref "/docs/usage/testing" >}}) for the search flags.
