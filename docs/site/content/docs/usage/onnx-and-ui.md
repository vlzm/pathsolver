---
title: "ONNX and the browser UI"
weight: 5
---

# ONNX and the browser UI

A trained model is small enough to ship to a browser and run the whole beam search
client-side. That is what `ui/web/cube/` does for the 3×3×3 cube.

## Exporting a checkpoint

```bash
python export_onnx.py --group_id 54 --model_id 333 --epoch 8192
```

`--group_id`, `--model_id` and `--epoch` are all required. `export_onnx.py` loads the
matching checkpoint from `weights/`, rebuilds the `Pilgrim` network and writes an ONNX
graph into `onnx/`. The exported graph's contract is `int64 [B, state_size] →
float [B]`: raw state vectors in — the one-hot encoding is part of the graph — and one
scalar score per state out, with a dynamic batch axis so the browser can size batches
to whatever the device handles.

You never pass architecture flags: `build_model` reads `hd1`, `hd2` and `nrd` out of
the checkpoint's own training log `logs/model_pXXX-tXXX_<model_id>.json`, so the
network always matches the weights.

| Flag | Default | Purpose |
|---|---|---|
| `--target_id` | `0` | Which solved state |
| `--out` | `onnx/pXXX-tXXX_<model_id>_e<epoch:05d>.onnx` | Output path |
| `--moves` | off | Also write `moves.json` (a copy of `generators/pXXX.json`) next to the model — this is what the UI loads |
| `--batch` | `8` | Dummy batch size used to trace the graph |

After exporting, the script runs the graph through `onnxruntime` on 64 random states
and prints the maximum absolute difference against PyTorch. It only prints — check the
number yourself; it should be around 10⁻⁶. If `onnxruntime` is not installed the check
is skipped with a message.

## The browser demo

`ui/web/cube/` is an interactive 3×3×3 solver that runs entirely in the browser:

- the cube widget is `cubing.js`'s `twisty-player`,
- inference runs on `onnxruntime-web` (WASM) against `model2.onnx`,
- the beam search is re-implemented in JavaScript in the shared
  `ui/web/solver/solver-core.js`, loaded by `worker.js` inside a **Web Worker** so the
  UI stays responsive; the cube's worker only supplies the config (beam width 2048,
  max 50 steps, batch size 128),
- the move permutations come from `moves.json` — byte-identical to
  `generators/p054.json`.

Alongside it, `ui/web/` holds several other standalone puzzle pages (sudoku, masyu,
bmf, …).

## Running locally

There is **no build step** — the pages are plain HTML/CSS/JS. Any static file server
works:

```bash
cd ui/web
python3 -m http.server 8765
```

Then open <http://127.0.0.1:8765/cube/>.

### Notes

- **Serve over HTTP.** Opening `index.html` as a `file://` URL fails: the browser's
  same-origin policy blocks the Web Worker and the `fetch` of `moves.json`.
- **First load needs network access.** The pages pull two dependencies from CDNs at
  runtime — `cubing.js` (the cube widget) and `onnxruntime-web` plus its WASM
  binaries.
- **Sudoku multiplayer is optional.** `ui/back/sudoku-back/` is a small Node backend
  needed only for that. Run it with `npm install && node server.js`, and set
  `ALLOWED_ORIGINS` to the origin serving the pages (defaults to
  `http://localhost:8080`).

## Deploying

Copy `ui/web/` to any web server — `ui/deploy/` holds the nginx config and a README
covering the public demo's setup. Since inference is client-side, no GPU or backend is involved in serving
the solver.
