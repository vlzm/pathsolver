---
math: true
title: "Architecture"
weight: 4
bookCollapseSection: true
mermaid: true
---

# Architecture

Pathsolver is a single-process PyTorch pipeline — there is no service split, no job
queue, no database. Everything is tensors on one device, with the filesystem as the
only interface between stages.

```mermaid
flowchart LR
  G[generators/pXXX.json<br/>permutations] --> T
  V[targets/*.pt<br/>solved state V0] --> T
  T[train.py<br/>random walks → PathSolver] --> W[weights/*.pt]
  W --> S[test.py<br/>batched beam search]
  D[datasets/*.pt<br/>scrambles] --> S
  G --> S
  V --> S
  S --> L[logs/test_*.json]
  W --> O[export_onnx.py → onnx/]
  O --> U[ui/web/cube<br/>in-browser solver]
```

Each stage is documented separately:

- **[Pipeline]({{< relref "/docs/architecture/pipeline" >}})** — the flow above, stage
  by stage, with the files each one reads and writes.
- **[The pathsolver package]({{< relref "/docs/architecture/pathsolver" >}})** — the ~1000
  lines that do the actual work: `model.py`, `trainer.py`, `searcher.py`, `dqn.py`,
  `utils.py`.
- **[Data contract]({{< relref "/docs/architecture/data-contract" >}})** — the naming
  and shape conventions that let a new puzzle drop in without touching code.

## Design invariants

Two properties are worth stating up front, because most of the design follows from
them:

1. **Nothing is puzzle-aware.** The model sees a one-hot state vector; the search
   applies permutations. A puzzle is data, never code.
2. **Nothing loops in Python over states.** Every operation — neighbour expansion,
   hashing, deduplication, scoring, pruning — is a batched tensor op. This is what
   allows beams of \( 2^{20} \) states and is where the speedup comes from.

Anything you add should preserve both.
