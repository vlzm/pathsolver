---
title: "Testing a pretrained model"
weight: 1
---

# Testing a pretrained model

The repository ships checkpoints in `weights/` for the 3×3×3, 4×4×4 and 5×5×5 cubes
(`model_id` `333`, `444`, `555`, epoch `8192`). These commands solve scrambles with
them directly — no training required.

## Cube 3×3×3 (UQTM metric)

```bash
python test.py --group_id 1 --target_id 0 --tests_num 3 --dataset santa \
  --num_steps 100 --verbose 1 --epoch 8192 --model_id 333 --B 262144 --device_id 0
```

## Cube 4×4×4 (UQTM metric)

```bash
python test.py --group_id 2 --target_id 0 --tests_num 3 --dataset santa \
  --num_steps 150 --verbose 1 --epoch 8192 --model_id 444 --B 262144 --device_id 0
```

## Cube 5×5×5 (UQTM metric)

```bash
python test.py --group_id 3 --target_id 0 --tests_num 3 --dataset santa \
  --num_steps 200 --verbose 1 --epoch 8192 --model_id 555 --B 524288 --device_id 0
```

## Cube 3×3×3 (QTM metric)

Group `054` is the same cube under the DeepCubeA move metric, which is what the
published optimality numbers are measured on:

```bash
python test.py --group_id 54 --target_id 0 --tests_num 3 --dataset deepcubea \
  --num_steps 100 --verbose 1 --epoch 8192 --model_id 333 --B 262144 --device_id 0
```

## Choosing the flags

- **`--tests_num`** — upper limit on how many scrambles to take from the start of the
  dataset. Use `--shift` to start further in.
- **`--dataset`** — which scramble set to solve:
  - `santa` — the official Kaggle Santa 2023 dataset.
  - `rnd` — 100 randomly generated scrambles (10 000 + 1 random steps from the solved
    state).
  - `deepcubea` (1000 scrambles), `deepcubeadifficult` (16), `deepcubeahard` (69) —
    **group `054` only**, for benchmarking against DeepCubeA and EfficientCube. (The
    README swaps these two counts; the sizes here are those of the shipped tensors —
    see [Data contract]({{< relref "/docs/architecture/data-contract" >}}).)
- **`--B`** — beam width. The main quality/compute knob: larger `B` means a higher
  solve rate and shorter solutions, at linear GPU-memory cost. If you hit OOM, halve
  it.
- **`--num_steps`** — maximum search depth. Must comfortably exceed the expected
  solution length, hence 100 for 3×3×3 and 200 for 5×5×5.
- **`--device_id`** — which GPU to use (default `0`).

Optimal solution lengths for the 16-scramble subset — shipped as
`deepcubeadifficult` — if you want to check optimality of your own run:

```text
[20, 20, 20, 21, 20, 20, 20, 20, 19, 20, 20, 19, 21, 20, 20, 20]
```

## Output

Results are written to
`logs/test_pXXX-tXXX-{dataset}_{model_id}_{epoch}_B{B}.json` — a JSON array in which
`moves` holds the generator indices leading from each scramble to the solved state.
See [Reference → Logs]({{< relref "/docs/reference/logs" >}}) for the full format and
`read-test-logs.py` for summarising a run.
