import json
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathsolver import Searcher, bellman_targets, generate_inverse_moves, get_neighbors

DEVICE = torch.device("cpu")


def load_group(group_id=34):
    """Load an LRX group (p034 = LRX with n=10) the same way train.py does."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(f"{root}/generators/p{group_id:03d}.json") as f:
        all_moves, move_names = json.load(f).values()
    all_moves = torch.tensor(all_moves, dtype=torch.int64, device=DEVICE)
    V0 = torch.load(f"{root}/targets/p{group_id:03d}-t000.pt", weights_only=True, map_location=DEVICE)
    return all_moves, move_names, V0


def apply_move(states, move):
    return torch.gather(states, 1, move.unsqueeze(0).expand(states.size(0), -1))


class ConstantModel(nn.Module):
    """Stub scoring model: value depends only on the state, no learned parameters."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, z):
        return self.fn(z).flatten()


def hamming_model(V0):
    """Hamming distance to the target - an admissible-ish heuristic for the searcher."""
    return ConstantModel(lambda z: (z != V0).sum(dim=1).float())


# --- generators -------------------------------------------------------------

def test_lrx_generators_are_involutive_and_inverse():
    all_moves, move_names, V0 = load_group()
    inverse_moves = generate_inverse_moves(move_names)
    states = V0.unsqueeze(0)

    for i, inv in enumerate(inverse_moves):
        # applying a move then its inverse returns the original state
        moved = apply_move(states, all_moves[i])
        assert torch.equal(apply_move(moved, all_moves[inv]), states), move_names[i]

    # X is a transposition, hence self-inverse: X^2 = id
    x_idx = move_names.index("X")
    assert torch.equal(apply_move(apply_move(states, all_moves[x_idx]), all_moves[x_idx]), states)


# --- get_neighbors ----------------------------------------------------------

def test_get_neighbors_matches_move_by_move_application():
    all_moves, _, V0 = load_group()
    states = torch.stack([V0, V0.flip(0)])

    neighbors = get_neighbors(states, all_moves)

    assert neighbors.shape == (2, all_moves.size(0), all_moves.size(1))
    for m in range(all_moves.size(0)):
        assert torch.equal(neighbors[:, m, :], apply_move(states, all_moves[m]))


def test_get_neighbors_batching_is_transparent():
    all_moves, _, V0 = load_group()
    states = V0.repeat(37, 1)
    assert torch.equal(get_neighbors(states, all_moves, batch_size=4),
                       get_neighbors(states, all_moves, batch_size=1024))


# --- bellman targets --------------------------------------------------------

def make_dqn_batch(n_states=64):
    all_moves, _, V0 = load_group()
    torch.manual_seed(0)
    # Random states plus the solved state, so the boundary condition is exercised.
    states = torch.stack([V0[torch.randperm(V0.numel())] for _ in range(n_states - 1)] + [V0.clone()])
    y_rw = torch.randint(1, 20, (n_states,), dtype=torch.int64)
    return all_moves, V0, states, y_rw


def test_bellman_targets_respect_random_walk_upper_bound():
    all_moves, V0, X, y_rw = make_dqn_batch()
    model = ConstantModel(lambda z: torch.full((z.size(0),), 100.0))

    y = bellman_targets(model, X, y_rw, all_moves, V0)

    solved = (X == V0).all(dim=1)
    # The true distance never exceeds the random walk length, so targets are clipped by it.
    assert (y[~solved] <= y_rw[~solved]).all()
    assert (y[~solved] >= 1).all()
    assert (y[solved] == 0).all()


def test_bellman_targets_apply_the_bellman_update():
    all_moves, V0, X, y_rw = make_dqn_batch()
    model = ConstantModel(lambda z: z[:, 0].float() / 100.0)

    y = bellman_targets(model, X, y_rw, all_moves, V0)

    neighbors = get_neighbors(X, all_moves)
    expected = 1.0 + model(neighbors.reshape(-1, X.size(1))).reshape(X.size(0), -1).min(dim=1).values
    expected = torch.clamp_min(torch.minimum(expected, y_rw.float()), 1.0)
    expected[(X == V0).all(dim=1)] = 0.0
    assert torch.allclose(y, expected, atol=1e-5)


def test_bellman_targets_rounding_is_optional():
    all_moves, V0, X, y_rw = make_dqn_batch()
    model = ConstantModel(lambda z: z[:, 0].float() / 100.0)

    y = bellman_targets(model, X, y_rw, all_moves, V0, flag_round=True)

    assert torch.equal(y, torch.round(y))


def test_bellman_targets_chunking_is_transparent():
    all_moves, V0, X, y_rw = make_dqn_batch()
    model = ConstantModel(lambda z: z[:, 0].float() / 100.0)
    assert torch.allclose(bellman_targets(model, X, y_rw, all_moves, V0, batch_size=8),
                          bellman_targets(model, X, y_rw, all_moves, V0, batch_size=4096))


# --- searcher ---------------------------------------------------------------

@pytest.mark.parametrize("n_scramble_moves", [1, 2, 3])
def test_searcher_solves_short_scrambles(n_scramble_moves):
    all_moves, _, V0 = load_group()
    torch.manual_seed(n_scramble_moves)

    state = V0.unsqueeze(0).clone()
    for m in torch.randint(0, all_moves.size(0), (n_scramble_moves,)):
        state = apply_move(state, all_moves[m])
    state = state.squeeze(0)

    searcher = Searcher(hamming_model(V0), all_moves, V0, device=DEVICE)
    solution, _ = searcher.get_solution(state, B=64, num_steps=10, num_attempts=1)

    assert solution is not None
    # Replaying the returned moves must actually reach the target.
    replayed = state.unsqueeze(0)
    for m in solution:
        replayed = apply_move(replayed, all_moves[m])
    assert torch.equal(replayed.squeeze(0), V0)


# --- trainer / dqn loop -----------------------------------------------------

def test_run_dqn_trains_and_saves_weights(tmp_path, monkeypatch):
    from pathsolver import PathSolver, Trainer

    all_moves, move_names, V0 = load_group()
    monkeypatch.chdir(tmp_path)

    model = PathSolver(state_size=V0.numel(), hd1=32, hd2=16, nrd=0, num_classes=V0.numel())
    trainer = Trainer(
        net=model, num_epochs=1, device=DEVICE, batch_size=256, name="test", K_min=1, K_max=5,
        all_moves=all_moves,
        inverse_moves=torch.tensor(generate_inverse_moves(move_names), dtype=torch.int64),
        V0=V0,
    )
    trainer.walkers_num = 20

    trainer.run_dqn(num_epochs=3, walkers_num=20)

    assert trainer.epoch == 3
    assert os.path.exists(f"weights/test_{trainer.id}_e00003.pth")
    assert os.path.exists(f"logs/dqn_test_{trainer.id}.csv")


def test_run_dqn_is_a_noop_for_zero_epochs(tmp_path, monkeypatch):
    from pathsolver import PathSolver, Trainer

    all_moves, move_names, V0 = load_group()
    monkeypatch.chdir(tmp_path)

    model = PathSolver(state_size=V0.numel(), hd1=32, hd2=16, nrd=0, num_classes=V0.numel())
    trainer = Trainer(
        net=model, num_epochs=1, device=DEVICE, batch_size=256, name="test", K_min=1, K_max=5,
        all_moves=all_moves,
        inverse_moves=torch.tensor(generate_inverse_moves(move_names), dtype=torch.int64),
        V0=V0,
    )

    trainer.run_dqn(num_epochs=0)

    assert trainer.epoch == 0
    assert not os.path.exists("weights") or os.listdir("weights") == []
