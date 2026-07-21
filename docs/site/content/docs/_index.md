---
title: "Documentation"
weight: 1
bookFlatSection: true
---

# Documentation

These docs cover Pathsolver end to end — from the problem it solves to every CLI flag.

## Sections

- **[Introduction]({{< relref "/docs/introduction" >}})** — what the project is, in one page.
- **[Research]({{< relref "/docs/research" >}})** — the problem formulation, the method
  (diffusion-distance regression + batched beam search) and the published results.
- **[Architecture]({{< relref "/docs/architecture" >}})** — how the repository is
  organised: the pipeline, the `pilgrim` package, and the data contract between
  `generators/`, `targets/` and `datasets/`.
- **[Usage]({{< relref "/docs/usage" >}})** — running tests with pretrained weights,
  training your own model, multi-agent evaluation, adding puzzles, ONNX export and the
  browser UI.
- **[Reference]({{< relref "/docs/reference" >}})** — the complete CLI flag list, the
  group ID table, the log format and a glossary.

## Suggested reading order

If you just want results, go
[Usage → Testing]({{< relref "/docs/usage/testing" >}}) and come back.
If you want to understand the method first, read
[Introduction]({{< relref "/docs/introduction" >}}) →
[Problem formulation]({{< relref "/docs/research/problem-formulation" >}}) →
[Method]({{< relref "/docs/research/method" >}}).
