// LRX puzzle: sort a permutation of 0..n-1 using three generators —
// L (cyclic shift left), R (cyclic shift right) and X (swap the first two positions).
// See arXiv:2502.18663; the solver model is trained by this repo's train.py.

// One entry per exported model. Files are fetched lazily on selection: each
// .onnx is ~2 MB, so we never load a model the user did not ask for.
const MODELS = {
  10: { modelUrl: './model-n10.onnx', movesUrl: './moves-n10.json' },
  15: { modelUrl: './model-n15.onnx', movesUrl: './moves-n15.json' }
};

const BEAM_WIDTH = 256;

// The generators are named L / L' / X / X' in generators/p03X.json (X' duplicates X).
// The paper and the buttons call the inverse shift R, so translate for display.
const DISPLAY_NAME = { "L": "L", "L'": "R", "X": "X", "X'": "X" };

// Button label -> generator name.
const BUTTON_MOVE = { "L": "L", "R": "L'", "X": "X" };

class LRXApp {
  constructor() {
    this.boardEl = document.getElementById('board');
    this.statusEl = document.getElementById('status');
    this.solutionEl = document.getElementById('solution');
    this.solveBtn = document.getElementById('solveBtn');
    this.sizeEl = document.getElementById('size');

    this.n = Number(this.sizeEl.value);
    this.state = [];
    this.moves = null;
    this.moveNames = null;
    this.isBusy = false;
    this.isInitialized = false;

    this.worker = new Worker('./worker.js');
    this.setupWorker();

    this.sizeEl.addEventListener('change', () => this.setSize(Number(this.sizeEl.value)));
    this.loadModel();
  }

  setupWorker() {
    this.worker.onmessage = (e) => {
      const { type, success, moves, moveNames, solution, step, total, error } = e.data;

      switch (type) {
        case 'init':
          if (success) {
            this.moves = moves;
            this.moveNames = moveNames;
            this.isInitialized = true;
            this.setBusy(false);
            this.reset();
            this.setStatus(`Ready — n = ${this.n}`);
          }
          break;

        case 'scramble':
          this.applyScramble(moves);
          break;

        case 'solved':
          this.applySolution(solution);
          break;

        case 'progress':
          this.solveBtn.textContent = `Step ${step}/${total}`;
          break;

        case 'error':
          this.setStatus('Error: ' + error, 'error');
          this.solveBtn.textContent = 'Solve';
          this.setBusy(false);
          break;
      }
    };
  }

  loadModel() {
    this.isInitialized = false;
    this.setBusy(true);
    this.setStatus(`Loading model for n = ${this.n}...`);
    this.hideSolution();
    this.worker.postMessage({
      type: 'init',
      data: {
        ...MODELS[this.n],
        solvedState: Array.from({ length: this.n }, (_, i) => i),
        beamWidth: BEAM_WIDTH,
        // Generous cap: the n=10 diameter is n(n-1)/2 = 45, and beam search needs
        // one step per move of the path it finds.
        maxSteps: this.n * (this.n - 1)
      }
    });
  }

  setSize(n) {
    if (this.isBusy || n === this.n) return;
    this.n = n;
    this.loadModel();
  }

  // ---- state -------------------------------------------------------------

  solvedState() {
    return Array.from({ length: this.n }, (_, i) => i);
  }

  // Moves act by gather, matching pilgrim/searcher.py and the ONNX input layout.
  applyMove(state, move) {
    return move.map(i => state[i]);
  }

  isSolved() {
    return this.state.every((v, i) => v === i);
  }

  // The conjectured longest element of LRX: swap the first two, reverse the rest.
  // For n=10 a full BFS confirms it sits at distance 45 = n(n-1)/2 (OEIS A186783).
  longestElement() {
    const p = this.solvedState();
    [p[0], p[1]] = [p[1], p[0]];
    let i = 2;
    while (i < this.n - i + 1) {
      [p[i], p[this.n - i + 1]] = [p[this.n - i + 1], p[i]];
      i++;
    }
    return p;
  }

  // ---- rendering ---------------------------------------------------------

  render(changed = []) {
    this.boardEl.innerHTML = '';
    this.boardEl.style.setProperty('--tile-count', this.n);
    this.state.forEach((value, i) => {
      const tile = document.createElement('div');
      tile.className = 'tile';
      if (value === i) tile.classList.add('placed');
      if (changed.includes(i)) tile.classList.add('moved');
      tile.textContent = value;
      this.boardEl.appendChild(tile);
    });
  }

  setState(state, previous = null) {
    this.state = state;
    const changed = previous ? state.map((v, i) => (v !== previous[i] ? i : -1)).filter(i => i >= 0) : [];
    this.render(changed);
  }

  setStatus(message, type = '') {
    this.statusEl.textContent = message;
    this.statusEl.className = `status ${type}`;
  }

  showSolution(moves) {
    const pretty = moves.map(m => DISPLAY_NAME[m] ?? m).join(' ');
    this.solutionEl.innerHTML = `<strong>Solution (${moves.length} moves):</strong>${pretty}`;
    this.solutionEl.classList.add('visible');
  }

  hideSolution() {
    this.solutionEl.classList.remove('visible');
  }

  setBusy(busy) {
    this.isBusy = busy;
    document.querySelectorAll('button').forEach(btn => { btn.disabled = busy; });
    this.sizeEl.disabled = busy;
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async playMoves(moveNames, delay) {
    for (const name of moveNames) {
      const idx = this.moveNames.indexOf(name);
      if (idx === -1) continue;
      const previous = this.state;
      this.setState(this.applyMove(previous, this.moves[idx]), previous);
      await this.sleep(delay);
    }
  }

  // ---- actions -----------------------------------------------------------

  move(label) {
    if (this.isBusy || !this.isInitialized) return;
    const idx = this.moveNames.indexOf(BUTTON_MOVE[label]);
    if (idx === -1) return;
    this.hideSolution();
    const previous = this.state;
    this.setState(this.applyMove(previous, this.moves[idx]), previous);
    this.setStatus(this.isSolved() ? 'Solved!' : 'Your move');
  }

  scramble() {
    if (this.isBusy || !this.isInitialized) return;
    this.setBusy(true);
    this.hideSolution();
    this.setStatus('Scrambling...');
    // A random walk this long mixes well past the diameter of the graph.
    this.worker.postMessage({ type: 'scramble', data: { numMoves: this.n * (this.n - 1) } });
  }

  async applyScramble(moves) {
    this.setState(this.solvedState());
    await this.playMoves(moves, 8);
    this.setStatus(`Scrambled with ${moves.length} random moves`);
    this.setBusy(false);
  }

  showLongest() {
    if (this.isBusy || !this.isInitialized) return;
    this.hideSolution();
    this.setState(this.longestElement());
    this.setStatus(`Longest element — needs ${this.n * (this.n - 1) / 2} moves`);
  }

  solve() {
    if (this.isBusy || !this.isInitialized) return;
    if (this.isSolved()) {
      this.setStatus('Already solved!', 'solved');
      return;
    }
    this.setBusy(true);
    this.hideSolution();
    this.solveBtn.textContent = 'Step 0';
    this.setStatus('Solving...', 'solving');
    this.worker.postMessage({ type: 'solve', data: { state: this.state } });
  }

  async applySolution(solution) {
    this.solveBtn.textContent = 'Solve';

    if (!solution) {
      this.setStatus('No solution found — try again or pick a smaller n', 'error');
      this.setBusy(false);
      return;
    }

    if (solution.length === 0) {
      this.setStatus('Already solved!', 'solved');
      this.setBusy(false);
      return;
    }

    this.showSolution(solution);
    this.setStatus('Solution found! Playing...');
    await this.playMoves(solution, 120);
    this.setStatus(`Solved in ${solution.length} moves!`, 'solved');
    this.setBusy(false);
  }

  reset() {
    if (this.isBusy || !this.isInitialized) return;
    this.setState(this.solvedState());
    this.hideSolution();
    this.solveBtn.textContent = 'Solve';
    this.setStatus('Reset - ready');
  }
}

window.app = new LRXApp();
