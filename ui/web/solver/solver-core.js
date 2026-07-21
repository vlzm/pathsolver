// Generic beam-search solver core, shared by the per-puzzle workers in ui/web/*/worker.js.
//
// A puzzle worker sets `self.SOLVER_CONFIG` and then importScripts() this file:
//
//   self.SOLVER_CONFIG = { modelUrl, movesUrl, solvedState, beamWidth, maxSteps, batchSize };
//   importScripts('../solver/solver-core.js');
//
// Relative URLs in the config resolve against the *worker's* URL (not this file's),
// so each puzzle keeps its model next to its own worker.js.
//
// The config may also be overridden per `init` message, which is how the LRX page
// swaps between models for different n without spawning a new worker.

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js');

// Configure WASM paths for ONNX Runtime
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';
ort.env.wasm.numThreads = 1; // Single thread to avoid CORS issues

class BeamSolver {
  constructor(config) {
    this.modelUrl = config.modelUrl;
    this.movesUrl = config.movesUrl;
    this.solvedState = Int32Array.from(config.solvedState);
    this.stateSize = this.solvedState.length;
    this.beamWidth = config.beamWidth ?? 2048;
    this.maxSteps = config.maxSteps ?? 50;
    this.batchSize = config.batchSize ?? 128;

    // Hash vector for state hashing
    this.hashVec = new BigInt64Array(this.stateSize);
    for (let i = 0; i < this.stateSize; i++) {
      this.hashVec[i] = BigInt(Math.floor(Math.random() * 1e15));
    }

    this.session = null;
    this.moves = null;
    this.moveNames = null;
    this.ready = false;
  }

  async init() {
    if (this.ready) return;
    this.session = await ort.InferenceSession.create(this.modelUrl);
    const response = await fetch(this.movesUrl);
    const data = await response.json();
    this.moves = data.actions.map(m => new Int32Array(m));
    this.moveNames = data.names;
    if (this.moves.length && this.moves[0].length !== this.stateSize) {
      throw new Error(
        `moves are ${this.moves[0].length}-long but solved state is ${this.stateSize}-long`
      );
    }
    this.ready = true;
  }

  simpleHash(state) {
    let hash = 0n;
    for (let i = 0; i < this.stateSize; i++) {
      hash += this.hashVec[i] * BigInt(state[i]);
    }
    return Number(hash % BigInt(Number.MAX_SAFE_INTEGER));
  }

  applyMove(state, move) {
    const result = new Int32Array(this.stateSize);
    for (let i = 0; i < this.stateSize; i++) {
      result[i] = state[move[i]];
    }
    return result;
  }

  isSolved(state) {
    for (let i = 0; i < this.stateSize; i++) {
      if (state[i] !== this.solvedState[i]) return false;
    }
    return true;
  }

  async batchPredict(states) {
    const batchSize = states.length;
    const inputData = new BigInt64Array(batchSize * this.stateSize);
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < this.stateSize; j++) {
        inputData[i * this.stateSize + j] = BigInt(states[i][j]);
      }
    }
    const tensor = new ort.Tensor('int64', inputData, [batchSize, this.stateSize]);
    const results = await this.session.run({ input: tensor });
    return Array.from(results.output.data);
  }

  quickselect(arr, k, left = 0, right = arr.length - 1) {
    while (right > left) {
      const pivotIndex = this.partition(arr, left, right);
      if (pivotIndex === k) break;
      if (pivotIndex < k) left = pivotIndex + 1;
      else right = pivotIndex - 1;
    }
  }

  partition(arr, left, right) {
    const pivot = arr[right].value;
    let i = left;
    for (let j = left; j < right; j++) {
      if (arr[j].value < pivot) {
        [arr[i], arr[j]] = [arr[j], arr[i]];
        i++;
      }
    }
    [arr[i], arr[right]] = [arr[right], arr[i]];
    return i;
  }

  async solve(initialState) {
    if (!this.ready) await this.init();
    const numMoves = this.moves.length;
    const nodes = [{parentId: -1, move: -1}];
    let beam = [{state: initialState, nodeId: 0}];

    for (let step = 0; step < this.maxSteps; step++) {
      postMessage({ type: 'progress', step: step + 1, total: this.maxSteps });

      const candidates = [];
      for (const item of beam) {
        for (let m = 0; m < numMoves; m++) {
          candidates.push({
            state: this.applyMove(item.state, this.moves[m]),
            parentNodeId: item.nodeId,
            move: m
          });
        }
      }

      const seen = new Map();
      const unique = [];
      for (const cand of candidates) {
        const hash = this.simpleHash(cand.state);
        if (!seen.has(hash)) {
          seen.set(hash, true);
          unique.push(cand);
        }
      }

      if (unique.length === 0) break;

      for (const u of unique) {
        u.nodeId = nodes.length;
        nodes.push({parentId: u.parentNodeId, move: u.move});
      }

      const values = [];
      for (let i = 0; i < unique.length; i += this.batchSize) {
        const batch = unique.slice(i, Math.min(i + this.batchSize, unique.length));
        const batchStates = batch.map(u => u.state);
        const batchVals = await this.batchPredict(batchStates);
        values.push(...batchVals);
      }

      for (let i = 0; i < unique.length; i++) {
        unique[i].value = values[i];
      }

      const k = Math.min(this.beamWidth, unique.length);
      this.quickselect(unique, k);
      const topK = unique.slice(0, k);

      for (const candidate of topK) {
        if (this.isSolved(candidate.state)) {
          return this.reconstructPath(nodes, candidate.nodeId);
        }
      }

      beam = topK.map(u => ({state: u.state, nodeId: u.nodeId}));
    }
    return null;
  }

  reconstructPath(nodes, solvedNodeId) {
    const path = [];
    let nodeId = solvedNodeId;
    while (nodes[nodeId].parentId !== -1) {
      path.unshift(nodes[nodeId].move);
      nodeId = nodes[nodeId].parentId;
    }
    return path.map(i => this.moveNames[i]);
  }

  createScrambledState(numMoves = 50) {
    let state = new Int32Array(this.solvedState);
    const scrambleMoves = [];
    for (let i = 0; i < numMoves; i++) {
      const moveIdx = Math.floor(Math.random() * this.moves.length);
      state = this.applyMove(state, this.moves[moveIdx]);
      scrambleMoves.push(this.moveNames[moveIdx]);
    }
    return { state, moves: scrambleMoves };
  }

  algToState(alg) {
    const moveStrs = alg.trim().split(/\s+/).filter(m => m);
    let state = new Int32Array(this.solvedState);
    for (const moveStr of moveStrs) {
      const idx = this.moveNames.indexOf(moveStr);
      if (idx !== -1) {
        state = this.applyMove(state, this.moves[idx]);
      }
    }
    return state;
  }
}

let solver = null;

// Handle messages from main thread
onmessage = async (e) => {
  const { type, data } = e.data;

  try {
    switch (type) {
      case 'init':
        // `data` overrides the worker's built-in config, so a page can switch
        // puzzle size by re-initialising instead of respawning the worker.
        solver = new BeamSolver({ ...self.SOLVER_CONFIG, ...(data || {}) });
        await solver.init();
        postMessage({
          type: 'init',
          success: true,
          moves: solver.moves.map(m => Array.from(m)),
          moveNames: solver.moveNames,
          solvedState: Array.from(solver.solvedState)
        });
        break;

      case 'scramble':
        const scramble = solver.createScrambledState(data.numMoves);
        postMessage({ type: 'scramble', moves: scramble.moves });
        break;

      case 'solve':
        // Callers may pass either a move sequence (`alg`) or a raw state.
        const currentState = data.state
          ? Int32Array.from(data.state)
          : solver.algToState(data.alg);
        if (solver.isSolved(currentState)) {
          postMessage({ type: 'solved', solution: [] });
        } else {
          const solution = await solver.solve(currentState);
          postMessage({ type: 'solved', solution });
        }
        break;
    }
  } catch (error) {
    postMessage({ type: 'error', error: error.message });
  }
};
