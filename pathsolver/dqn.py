import torch
from .utils import get_neighbors


@torch.no_grad()
def bellman_targets(net, X, y_rw, all_moves, V0, batch_size=2**14, flag_round=False):
    """Modified DQN targets for a batch of states.

    Bellman update on the Cayley graph: V(g) = 1 + min_{neighbors} V(neighbor).
    The random walk labels y_rw are an upper bound on the true distance, so the
    Bellman estimate is clipped against them; distances are at least 1 everywhere
    except the solved state itself.

    :param net: PathSolver model (scalar regressor)
    :param X: Tensor [N, state_size] of states
    :param y_rw: Tensor [N] of random walk lengths (upper bound on distance)
    :param all_moves: Tensor [n_gens, state_size] of permutations
    :param V0: Tensor [state_size], the solved state
    :param flag_round: round targets to integers (no improvement per the paper)
    :return: Tensor [N] of float targets
    """
    net.eval()
    n_gens, state_size = all_moves.size(0), all_moves.size(1)
    y = torch.empty(X.size(0), dtype=torch.float32, device=X.device)

    # Neighbors are expanded per chunk so that the model always sees ~batch_size rows.
    chunk = max(1, batch_size // n_gens)
    for i in range(0, X.size(0), chunk):
        batch_states = X[i:i + chunk]
        neighbors = get_neighbors(batch_states, all_moves, batch_size)
        preds = net(neighbors.reshape(-1, state_size)).reshape(batch_states.size(0), n_gens)
        y[i:i + chunk] = 1.0 + preds.min(dim=1).values.float()

    # Restrictions coming from the diffusion distance: true distance <= random walk length.
    y = torch.minimum(y, y_rw.to(y.dtype))
    y = torch.clamp_min(y, 1.0)

    # Boundary condition - at the origin the distance is zero.
    y[(X == V0).all(dim=1)] = 0.0

    if flag_round:
        y = torch.round(y)

    return y
