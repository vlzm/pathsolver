// LRX Solver Web Worker.
// The beam search lives in ../solver/solver-core.js, shared with the cube page.
// The per-n configuration (model, generators, solved state) is owned by script.js
// and arrives with the `init` message, since the page lets the user switch n.

self.SOLVER_CONFIG = {
  batchSize: 128
};

importScripts('../solver/solver-core.js');
