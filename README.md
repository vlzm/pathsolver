# Zero-Knowledge ML for Finding Short Paths on Large Cayley Graphs

[![Documentation](https://img.shields.io/badge/docs-pathsolver-2ea44f.svg)](https://vlzm.github.io/pathsolver/)
[![NeurIPS 2025 Spotlight](https://img.shields.io/badge/NeurIPS%202025-Spotlight-0b5fff.svg)](#)
[![arXiv:2502.13266](https://img.shields.io/badge/arXiv-2502.13266-b31b1b.svg)](https://www.arxiv.org/pdf/2502.13266)
[![arXiv:2502.18663](https://img.shields.io/badge/arXiv-2502.18663-b31b1b.svg)](https://arxiv.org/abs/2502.18663)

📖 **Full documentation:** <https://vlzm.github.io/pathsolver/> — pipeline overview, architecture of the `pathsolver` package, usage guides, and the research write-up.

**Lightweight, end-to-end PyTorch pipeline** — data generation → training → search — all in `torch`, optimized for massive GPU parallelism. This repository is the official implementation of the NeurIPS 2025 Spotlight paper *A Machine Learning Approach That Beats Large Rubik's Cubes* ([arXiv:2502.13266](https://arxiv.org/abs/2502.13266)): a novel ML approach for finding short paths on large **Cayley graphs** (e.g. across permutation puzzles such as the Rubik’s Cube) by training neural networks to estimate diffusion distances and using those predictions to guide an efficient beam search. This approach is **zero human knowledge** (no handcrafted heuristics or domain rules).

The repository also implements the follow-up paper in the CayleyPy series, *CayleyPy RL: Pathfinding and Reinforcement Learning on Cayley Graphs* ([arXiv:2502.18663](https://arxiv.org/abs/2502.18663)): the optional **Modified DQN** refinement of the value estimates (see [Training](#training-your-model)) and the **LRX** puzzle family (cyclic shift + transposition generators, groups `034`+) studied there.

<p align="center">
  <img src="assets/fig.png" alt="Architecture: diffusion-distance predictor + batched beam search" width="720">
</p>

The same PyTorch pipeline works across distinct **Cayley graphs** without changing model architecture and hyperparameters. With one NVIDIA H100, model parameters (N₁ = 1024, N₂ = 256, Nᵣ = 1), beam width 2²⁰ and only 1.28×10⁸ training examples we achieve these results:

<p align="center">

| Puzzle            | Graph size        | Training (min) | Solving (min) | Avg. solution len. | Solved (%)     |
|:------------------|:------------------:|---------------:|--------------:|------------------:|---------------:|
| Rubik's Cube 3×3×3        | 4×10¹⁹            | 0.8            | 0.17          | 20.5 ± 0.1         | 100            |
| Rubik's Cube 4×4×4        | 7×10⁴⁵            | 1.5            | 1.24          | 65.0 ± 1.0         | 94             |
| Pancake Graph 55          | 3×10⁷³            | 1.4            | 3.49          | 50.0 ± 0.3         | 100            |
| Klotski 6×6 (PBC)         | 2×10⁴¹            | 1.2            | 0.45          | 113 ± 2.0          | 100            |

</p>

Using more computations the method reaches 98% optimality on 3×3×3 (QTM) and outperforms the top Santa Kaggle 2023 submissions on 3×3×3, 4×4×4 and 5×5×5 in average solution length. On 3×3×3, the batched GPU beam search delivers **20× faster** path finding than the previous state of the art at comparable solution lengths.


## Testing a Pre-trained Model

Run these commands to solve puzzles using preloaded model weights:

**Cube 3x3x3 (UQTM metric)**

```bash
python test.py --group_id 1 --target_id 0 --tests_num 3 --dataset santa --num_steps 100 --verbose 1 --epoch 8192 --model_id 333 --B 262144 --device_id 0
```

**Cube 4x4x4 (UQTM metric)**

```bash
python test.py --group_id 2 --target_id 0 --tests_num 3 --dataset santa --num_steps 150 --verbose 1 --epoch 8192 --model_id 444 --B 262144 --device_id 0
```

**Cube 5x5x5 (UQTM metric)**

```bash
python test.py --group_id 3 --target_id 0 --tests_num 3 --dataset santa --num_steps 200 --verbose 1 --epoch 8192 --model_id 555 --B 524288 --device_id 0
```

**Cube 3x3x3 (QTM metric)**

```bash
python test.py --group_id 54 --target_id 0 --tests_num 3 --dataset deepcubea --num_steps 100 --verbose 1 --epoch 8192 --model_id 333 --B 262144 --device_id 0
```

### Notes

* `tests_num` — sets an upper limit on the number of scrambles to test from the beginning of the dataset.
* `dataset` — selects the dataset to use:
  * `santa` — official Kaggle Santa 2023 dataset.
  * `rnd` — randomly generated dataset (100 scrambles) using 10,000(+1) random steps from the solved state.
  * `deepcubea` (1000 DeepCubeA scrambles), `deepcubeadifficult` (16 DeepCubeA subset scrambles), `deepcubeahard` (69 DeepCubeA subset scrambles) — available only for group `054` (3x3x3, QTM metric), useful for benchmarking against DeepCubeA and EfficientCube.
* `device_id` — specifies which GPU to use for testing (default is `0`). Useful when multiple GPUs are available.

Optimal solution lengths for the scrambles in the `deepcubeadifficult`: `[20, 20, 20, 21, 20, 20, 20, 20, 19, 20, 20, 19, 21, 20, 20, 20]`.

## Output

Test results saved in `logs/test_pXXX-tXXX-{dataset}_{model_id}_{epoch}_{B}.json`. Each file is a JSON array where `moves` contain generator indices used to reach the solved state for corresponding scramble.

## Training Your Model

Run training using your puzzle configuration:

```bash
python train.py --group_id <id> --target_id 0 --epochs <epochs> --hd1 <N_1> --hd2 <N_2> --nrd <N_r> --batch_size 10000 --K_max <K_max> --device_id 0
```

Example for 24-puzzle:

```bash
python train.py --group_id 28 --target_id 0 --epochs 16 --hd1 1024 --hd2 512 --nrd 1 --batch_size 10000 --K_max 100 --device_id 0
```

After training, use the assigned model_id (generated as int(time.time()) during training) to run beam search evaluation to use trained model:

```bash
python test.py --group_id 28 --target_id 0 --tests_num 3 --dataset rnd --num_steps 300 --num_attempts 1 --verbose 1 --epoch 16 --model_id {MODEL_ID} --B 65536 --device_id 0
```

Replace `{MODEL_ID}` with the actual numeric identifier saved in the logs.

### Modified DQN refinement

After the random-walk phase you can optionally refine the value estimates with a Bellman update on the Cayley graph, following the *CayleyPy RL* paper ([arXiv:2502.18663](https://arxiv.org/abs/2502.18663)):

```bash
python train.py --group_id 34 --target_id 0 --epochs 30 --epochs_dqn 200 --K_max 50 --device_id 0
```

* `--epochs_dqn` — number of Modified DQN epochs (`0` disables the phase).
* `--dqn_walkers` — walkers per DQN epoch (`0` means 1/10 of the warm-up walkers).
* `--dqn_round` — round DQN targets to integers (reported in the paper as giving no improvement; kept for reproduction).

Run this **on top of** a diffusion-distance warm-up, not instead of one: the paper reports that DQN epochs reliably improve the warm-up model (shorter solutions, higher solve rate) but by a modest margin, while DQN from scratch performs poorly.

## Adding New Puzzle Groups

To add your puzzle:

1. Define your puzzle moves in `generators` folder as `pXXX.json`.
2. Place a torch tensor `.pt` (1D tensor) in `targets`.
3. Add scrambles dataset (2D tensor, each row as a scramble) to `datasets` folder.

## Multi-Agent Evaluation

For **Cube 3x3x3 (QTM)** on the **`deepcubea`** dataset, a multi-agent script `traintest-multiagent.sh` automates:

* training multiple agents (`group_id=054`, `target_id=0`)
* testing each agent on the same scrambles
* aggregating results: per-agent and ensemble statistics

### What It Does

After training and testing `A` agents, the script calls `read-test-logs-multiagent.py` to compute:

* average solution length per agent
* ensemble stats (shortest solution per scramble)
* solved percentage
* move sequences from the best agent per scramble

### Output Example

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

All logs are saved in `logs/`, with results printed at the end.

## Available Groups and Kmax

| Group ID | Puzzle                        | Kmax used |
| -------- | ----------------------------- | --------- |
| 000      | Cube 2x2x2                    | 15        |
| 001      | Cube 3x3x3                    | 26        |
| 002      | Cube 4x4x4                    | 45        |
| 003      | Cube 5x5x5                    | 65        |
| 004      | Cube 6x6x6                    | 150       |
| 011      | Wreath 6/6                    | 10        |
| 012      | Wreath 7/7                    | 10        |
| 013      | Wreath 12/12                  | 20        |
| 014      | Wreath 21/21                  | 35        |
| 015      | Wreath 33/33                  | 75        |
| 017      | Globe 1/8                     | 60        |
| 018      | Globe 1/16                    | 110       |
| 019      | Globe 2/6                     | 25        |
| 020      | Globe 3/4                     | 40        |
| 021      | Globe 6/4                     | 40        |
| 022      | Globe 6/8                     | 165       |
| 023      | Globe 6/10                    | 170       |
| 024      | Globe 3/33                    | 500       |
| 025      | Globe 8/25                    | 700       |
| 026      | Puzzle 8                      | 30        |
| 027      | Puzzle 15                     | 80        |
| 028      | Puzzle 24                     | 150       |
| 029      | Puzzle 35                     | 200       |
| 030      | Puzzle 48                     | 250       |
| 031      | Puzzle 63                     | 300       |
| 034      | LRX 10                        | 50        |
| 035      | LRX 15                        | 100       |
| 036      | LRX 20                        | 200       |
| 037      | LRX 25                        | 300       |
| 044      | Pancake 10                    | 15        |
| 045      | Pancake 15                    | 25        |
| 046      | Pancake 20                    | 30        |
| 047      | Pancake 25                    | 40        |
| 048      | Pancake 30                    | 45        |
| 049      | Pancake 35                    | 55        |
| 050      | Pancake 40                    | 60        |
| 051      | Pancake 45                    | 65        |
| 052      | Pancake 50                    | 70        |
| 053      | Pancake 55                    | 75        |
| 054      | Cube 3x3x3 (DeepCubeA metric) | 26        |

The `LRX` groups are the Cayley graph of the symmetric group with generators L/R (cyclic shift left/right) and X (transposition of the first two positions), studied in the *CayleyPy RL* paper ([arXiv:2502.18663](https://arxiv.org/abs/2502.18663)); generators and targets for larger sizes up to LRX 55 (groups `038`–`043`) ship in the repository as well.

For reproducing results from Table 4, refer to provided `traintest-tab4-santa.sh` and `traintest-tab4-rnd.sh`, paper/solver-scrambles and paper/figure-scrambles contain generators and scrambles used in each specific experiment. The classical 15-puzzle (no wrap-around) is not a Cayley graph: node degrees vary with the blank position, hence the state graph is not vertex-transitive. We include it as a non-Cayley baseline at separate branch [puzzle-15](https://github.com/khoruzhii/cayleypy-cube/tree/puzzle-15).

## Browser UI

The `ui/` folder contains a browser demo that runs a trained model entirely
client-side: `ui/web/cube/` is an interactive 3x3x3 solver where the beam search
runs in a Web Worker on top of `onnxruntime-web`, using the exported network in
`model2.onnx` and the move permutations in `moves.json`. `ui/web/lrx/` is the same
setup for the LRX permutation puzzle from arXiv:2502.18663, with ONNX models for
n = 10 and n = 15. Alongside them `ui/web/`
holds several other standalone puzzle pages (sudoku, masyu, bmf, ...).

**Live demo:** the UI is published to GitHub Pages automatically, so you can try
it without installing anything — the 3×3×3 solver runs at
<https://vlzm.github.io/pathsolver/ui/cube/>.

There is **no build step** — the pages are plain HTML/CSS/JS. To run them locally
you only need a static file server:

```bash
cd ui/web
python3 -m http.server 8765
```

Then open <http://127.0.0.1:8765/cube/>.

### Notes

* Serve over HTTP — opening `index.html` as a `file://` URL fails, because the
  Web Worker and the `fetch` of `moves.json` are blocked by the browser's
  same-origin policy.
* The pages load two dependencies from CDNs at runtime: `cubing.js` (the
  `twisty-player` cube widget) and `onnxruntime-web` (plus its WASM binaries),
  so the first load needs network access.
* `ui/back/sudoku-back/` is an optional Node backend, needed only for Sudoku
  multiplayer. Run it with `npm install && node server.js`, and set
  `ALLOWED_ORIGINS` to the origin serving the pages (defaults to
  `http://localhost:8080`). This backend is **not** part of the GitHub Pages
  deployment, so multiplayer is unavailable there (the single-player solver
  still works).
* **Deployment is automatic.** The docs workflow
  (`.github/workflows/docs.yml`) copies `ui/web/` into the published site on
  every push to `main`, serving it under `…/pathsolver/ui/`. To host it
  elsewhere, just copy `ui/web/` to any static web server (see `ui/deploy/` for
  an nginx example) — inference is client-side, so no GPU or backend is needed.

## Citation

If you use this work in research, please cite one of the following BibTeX entries.

```bibtex
% Proceedings version (preferred once the NeurIPS paper is out)
@inproceedings{Chervov2025NeurIPSRubiks,
  author    = {Alexander Chervov and Kirill Khoruzhii and Nikita Bukhal and Jalal Naghiyev and Vladislav Zamkovoy and Ivan Koltsov and Lyudmila Cheldieva and Arsenii Sychev and Arsenii Lenin and Mark Obozov and Egor Urvanov and Alexey M. Romanov},
  title     = {A machine learning approach that beats Rubik’s cubes},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year      = {2025},
  note      = {Spotlight},
  url       = {https://neurips.cc/}
}
```

```bibtex
% arXiv preprint (use this if you are citing the preprint)
@misc{Chervov2025arXivRubiks,
  author        = {Alexander Chervov and Kirill Khoruzhii and Nikita Bukhal and Jalal Naghiyev and Vladislav Zamkovoy and Ivan Koltsov and Lyudmila Cheldieva and Arsenii Sychev and Arsenii Lenin and Mark Obozov and Egor Urvanov and Alexey M. Romanov},
  title         = {A Machine Learning Approach That Beats Large Rubik's Cubes},
  year          = {2025},
  eprint        = {2502.13266},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2502.13266}
}
```

If you use the Modified DQN refinement or the LRX puzzles, please also cite the follow-up paper:

```bibtex
% CayleyPy RL: Modified DQN refinement and the LRX study
@misc{Chervov2026CayleyPyRL,
  author        = {A. Chervov and M. Obozov and A. Soibelman and S. Lytkin and I. Kiselev and S. Fironov and A. Lukyanenko and A. Dolgorukova and A. Ogurtsov and F. Petrov and S. Krymskii and M. Evseev and L. Grunvald and D. Gorodkov and G. Antiufeev and G. Verbii and V. Zamkovoy and L. Cheldieva and I. Koltsov and A. Sychev and A. Eliseev and S. Nikolenko and N. Narynbaev and R. Turtayev and N. Rokotyan and S. Kovalev and A. Rozanov and V. Nelin and S. Ermilov and L. Shishina and D. Mamayeva and A. Korolkova and K. Khoruzhii and A. Romanov},
  title         = {CayleyPy RL: Pathfinding and Reinforcement Learning on Cayley Graphs},
  year          = {2026},
  eprint        = {2502.18663},
  archivePrefix = {arXiv},
  doi           = {10.4310/ATMP.260413005111},
  url           = {https://arxiv.org/abs/2502.18663}
}
```
