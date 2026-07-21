---
title: "Multi-agent evaluation"
weight: 3
---

# Multi-agent evaluation

Independently trained models fail on different scrambles and find different-length
solutions on the ones they share. Taking the shortest solution per scramble across
several agents is therefore a cheap, embarrassingly parallel improvement to both solve
rate and average length — this is the *ensemble* setting reported in the paper.

`traintest-multiagent.sh` automates it for **Cube 3×3×3 (QTM)** — `group_id=054`,
`target_id=0` — on the `deepcubea` dataset:

```bash
./traintest-multiagent.sh                    # defaults: A=2 TESTS_NUM=4 EPOCH=16 B=65536
./traintest-multiagent.sh 8 128 128 1048576  # A TESTS_NUM EPOCH B
A=8 EPOCH=128 ./traintest-multiagent.sh      # same knobs as environment variables
```

## What it does

For `A` agents, the script:

1. trains each agent independently — no seed is set anywhere, so the agents differ by
   random weight initialisation alone, and each gets its own
   `model_id = int(time.time())`,
2. tests every agent on the **same** scrambles,
3. calls `read-test-logs-multiagent.py` to aggregate.

The aggregation reports:

- average solution length per agent,
- ensemble statistics — the shortest solution per scramble across all agents,
- solved percentage,
- the move sequence from the best agent for each scramble.

> **Fix needed before the script runs end to end.** Its last line invokes
> `python scripts/read-test-logs-multiagent.py`, but there is no `scripts/` directory —
> the file sits at the repository root. Drop the `scripts/` prefix, or run the
> aggregation by hand as shown below.

The defaults (`A=2`, `TESTS_NUM=4`, `EPOCH=16`, `B=65536`) are a smoke test — four
scrambles at a small beam width, enough to check the plumbing. For numbers worth
reporting, raise all four; the aggregator's own defaults (`EPOCH=128`, `B=1048576`)
indicate the intended scale.

## Output example

The columns below are the ones the aggregator prints; the numbers are illustrative,
not from a recorded run.

```text
=== per agent ===
          tests  solved_%  avg_len
123456789   1000     97.3     21.4
123456790   1000     98.2     20.8

=== ensemble (shortest per scramble) ===
solved %           : 99.1
avg solution length: 19.95

=== moves (winning agent) ===
 test_num  solution_length   model_id            moves
        0                20  123456790  [2, 0, 4, 5, 1, ...]
        ...
```

The point of the ensemble is that the agents' failures are largely uncorrelated, so
both solve rate and average length improve over any single agent.

All logs are written to `logs/` as usual, and the summary is printed at the end. To
re-aggregate an existing set of runs without retraining, call the script directly —
passing the same `A`, `EPOCH` and `B` you tested with, since it locates log files by
name:

```bash
python read-test-logs-multiagent.py 2 16 65536   # matches the shell script's defaults
```

With no arguments it looks for `_128_B1048576` logs, which the shell script does not
produce at its default settings; you would just get "not found" warnings.

See [Reference → Logs]({{< relref "/docs/reference/logs" >}}) for how it locates and
groups the per-agent log files.
