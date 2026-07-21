---
title: "Pipeline"
weight: 1
math: true
---

# Pipeline

The five stages, in the order data flows through them.

## 1. Puzzle definition

A puzzle is three files, no code:

| File | Content |
|---|---|
| `generators/pXXX.json` | `{"actions": [[...], ...], "names": ["R", "R'", ...]}` — one permutation per legal move |
| `targets/*.pt` | 1-D `torch` tensor: the solved state \( V_0 \) |
| `datasets/pXXX-tXXX-{rnd,santa}.pt` | 2-D tensor, one scramble per row |

Details and naming rules in
[Data contract]({{< relref "/docs/architecture/data-contract" >}}).

## 2. Data generation (on the fly)

There is no dataset-building step for training — `Trainer.generate_random_walks`
produces training data inside the training loop, on the GPU. Per epoch it launches
`walkers_num = 1_000_000 // K_max` walkers for each length
\( K \in [K_{\min}, K_{\max}] \), so an epoch is ~10⁶ labelled states regardless of
puzzle. Walks avoid immediately undoing the previous move
(`inverse_moves`, built by `generate_inverse_moves` from the generator names).

Because generation is GPU-side and fresh every epoch, the model never sees the same
example twice and nothing is written to disk.

## 3. Training

`train.py` assembles the pieces, then hands off to `Trainer.run`:

- builds `Pilgrim(state_size, hd1, hd2, nrd, dropout_rate)`,
- optionally warm-starts from `--weights <name>` in `weights/`,
- runs `--epochs` epochs of MSE regression on random-walk labels,
- optionally runs `--epochs_dqn` epochs of Bellman refinement
  ([`pilgrim/dqn.py`]({{< relref "/docs/architecture/pilgrim" >}})),
- writes checkpoints to `weights/` and a training log to `logs/`.

The model identity is `model_id = int(time.time())`, assigned in `Trainer.__init__`
and printed at start — you need it to load the checkpoint back in step 4.

## 4. Beam search evaluation

`test.py` loads a checkpoint (`--model_id` + `--epoch`) and a scramble set
(`--dataset`), and runs `Searcher.get_solution` per scramble with beam width `--B`, up to
`--num_steps` depth and `--num_attempts` restarts. Each solved scramble yields the
list of generator indices leading to \( V_0 \).

Output goes to `logs/test_pXXX-tXXX-{dataset}_{model_id}_{epoch}_B{B}.json` — note the
literal `B` before the beam width, and the `_shift<N>` / `_skip<list>` suffixes appended
when those flags are used
([format]({{< relref "/docs/reference/logs" >}})). With `--return_tree 1` the full
search tree is dumped to `forest/` for inspection.

## 5. Export and browser inference

`export_onnx.py` converts a checkpoint into `onnx/` — a graph whose contract is
`int64 [B, state_size] → float [B]`: raw state vectors in (one-hot encoding happens
inside the graph), scalar scores out. `ui/web/cube/` then re-implements the beam
search in JavaScript on top of `onnxruntime-web`, running inside a Web Worker, so the
3×3×3 solver runs **entirely client-side** with no server.

See [ONNX and the browser UI]({{< relref "/docs/usage/onnx-and-ui" >}}).
