// Rubik's Cube Solver Web Worker.
// The beam search itself lives in ../solver/solver-core.js and is shared with the
// other puzzles; this file only pins the cube-specific configuration.

self.SOLVER_CONFIG = {
  modelUrl: './model2.onnx',
  movesUrl: './moves.json',
  solvedState: [
    0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2,
    3,3,3,3,3,3,3,3,3, 4,4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5,5
  ],
  beamWidth: 2048,
  maxSteps: 50,
  batchSize: 128
};

importScripts('../solver/solver-core.js');
