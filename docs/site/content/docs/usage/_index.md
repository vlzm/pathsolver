---
title: "Usage"
weight: 5
bookCollapseSection: true
---

# Usage

Practical guides, in the order you are likely to need them.

- **[Testing a pretrained model]({{< relref "/docs/usage/testing" >}})** — solve
  3×3×3 / 4×4×4 / 5×5×5 with the released weights. Start here.
- **[Training]({{< relref "/docs/usage/training" >}})** — train your own model and
  evaluate it.
- **[Multi-agent evaluation]({{< relref "/docs/usage/multiagent" >}})** — train an
  ensemble and aggregate per-agent and ensemble statistics.
- **[Adding new puzzles]({{< relref "/docs/usage/new-puzzles" >}})** — bring your own
  Cayley graph.
- **[ONNX and the browser UI]({{< relref "/docs/usage/onnx-and-ui" >}})** — export a
  checkpoint and run it client-side.

## Requirements

Python with PyTorch and a CUDA GPU (`pip install -r requirements.txt`). Everything
runs on CPU too, but beam widths beyond a few thousand are impractical there. Pick the
device with `--device_id` when several GPUs are available.

Every flag accepted by `train.py` and `test.py` is listed in
[Reference → CLI]({{< relref "/docs/reference/cli" >}}).
