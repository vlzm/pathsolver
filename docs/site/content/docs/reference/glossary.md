---
title: "Glossary"
weight: 4
math: true
---

# Glossary

**Beam search** — search that keeps only the \( B \) most promising states at each
depth, discarding the rest. Incomplete (it can miss the optimal path) but bounded in
memory. Here every step is a batched GPU tensor operation; see
[Method]({{< relref "/docs/research/method" >}}).

**Beam width \( B \)** — how many states survive each step (`--B`). The main
quality/compute knob: larger \( B \) gives higher solve rates and shorter solutions at
linear memory cost. Published results use up to \( 2^{20} \).

**Bellman update** — refining a value estimate with
\( V(g) \leftarrow 1 + \min_{s} V(gs) \), used in the optional Modified DQN phase
(`pilgrim/dqn.py`, `--epochs_dqn`).

**Cayley graph** — graph on the elements of a group \( G \) with edges \( g \to gs \)
for generators \( s \in S \). Vertex-transitive, defined implicitly by \( |S| \)
permutations. See [Problem formulation]({{< relref "/docs/research/problem-formulation" >}}).

**DeepCubeA** — prior deep-RL Rubik's solver (value iteration + weighted A\*). Its
scramble sets ship here as the `deepcubea*` datasets on group `054`, and it is the main
speed and optimality baseline.

**Diffusion distance** — what the network actually learns: the expected number of
random steps separating a state from the target, approximated by regressing on
random-walk lengths. Monotone in the true distance and cheap to evaluate, which is all
the beam search needs for ranking.

**EfficientCube** — another learned-heuristic Rubik's solver used as a baseline.

**Generators** — the legal moves, as permutations of the state vector. Stored in
`generators/pXXX.json`; must be closed under inversion.

**God's number** — the diameter of the graph: the largest possible optimal solution
length. 20 for 3×3×3 in HTM, 26 in QTM.

**Group ID** — integer identifying a puzzle, written `pXXX` in filenames. See
[Groups]({{< relref "/docs/reference/groups" >}}).

**\( K_{\max} \) / \( K_{\min} \)** — the random-walk length range used to generate
training data (`--K_max`, `--K_min`). \( K_{\max} \) must exceed the graph's diameter;
it is the one genuinely per-puzzle hyperparameter.

**Klotski** — sliding block puzzle; the 6×6 variant with periodic boundary conditions
(PBC) is one of the four headline benchmarks.

**`model_id`** — a run's identity, set to `int(time.time())` when training starts.
Needed to load a checkpoint back.

**Optimality** — the fraction of solutions whose length equals the known optimal
length for that scramble. The method reaches 98% on 3×3×3 QTM.

**Pancake graph** — Cayley graph whose generators reverse a prefix of the
permutation. Pancake 55 has \( 3 \times 10^{73} \) vertices.

**QTM (quarter-turn metric)** — move metric counting only 90° face turns; a 180° turn
costs two moves. Used by DeepCubeA and by group `054`.

**Scramble** — a state reached from the solved state by random moves; the input to be
solved. Stored as rows of `datasets/*.pt`.

**Solve rate** — percentage of scrambles solved within the step and beam budget.

**State size** — length of the state vector, i.e. the number of labelled positions.
With `num_classes` it determines the model's one-hot input width.

**UQTM** — the move metric used by the Kaggle Santa 2023 puzzles, corresponding to
groups `001`–`004`.

**Vertex-transitive** — every vertex of the graph looks identical from its own
perspective. True of Cayley graphs; this is what lets any path-finding problem reduce
to solving a single scrambled state.

**\( V_0 \)** — the solved (target) state, a 1-D tensor from `targets/*.pt`.

**Zero-knowledge** — no handcrafted heuristics, domain rules or pattern databases; the
only inputs are the generators and the target state.
