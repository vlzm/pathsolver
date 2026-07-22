import torch
import json

def load_cube_data(cube_size, cube_type, device):
    """Load cube data based on cube size and type (qtm or all)."""
    file_path = f"generators/{cube_type}_cube{cube_size}.json"
    
    with open(file_path, 'rb') as f:
        data = json.load(f)
    
    actions = data["actions"]
    action_names = data["names"]
    
    return torch.tensor(actions, dtype=torch.int64, device=device), action_names

def generate_inverse_moves(moves):
    """Generate the inverse moves for a given list of moves."""
    inverse_moves = [0] * len(moves)
    for i, move in enumerate(moves):
        if "'" in move:  # It's an a_j'
            inverse_moves[i] = moves.index(move.replace("'", ""))
        else:  # It's an a_j
            inverse_moves[i] = moves.index(move + "'")
    return inverse_moves

def get_neighbors(states, all_moves, batch_size=2**14):
    """Return neighboring states for each state in the batch.

    :param states: Tensor [N, state_size]
    :param all_moves: Tensor [n_gens, state_size] of permutations
    :return: Tensor [N, n_gens, state_size]
    """
    n_gens, state_size = all_moves.size(0), all_moves.size(1)
    neighbors = torch.empty(states.size(0), n_gens, state_size, device=states.device, dtype=states.dtype)

    for i in range(0, states.size(0), batch_size):
        batch_states = states[i:i + batch_size]
        neighbors[i:i + batch_size] = torch.gather(
            batch_states.unsqueeze(1).expand(batch_states.size(0), n_gens, state_size),
            2,
            all_moves.unsqueeze(0).expand(batch_states.size(0), n_gens, state_size)
        )
    return neighbors

def state2hash(states, hash_vec, batch_size=2**14):
    """Convert states to hashes."""
    num_batches = (states.size(0) + batch_size - 1) // batch_size
    result = torch.empty(states.size(0), dtype=torch.int64, device=states.device)
    
    for i in range(num_batches):
        batch = states[i * batch_size:(i + 1) * batch_size].to(torch.int64)
        batch_hash = torch.sum(hash_vec * batch, dim=1)
        result[i * batch_size:(i + 1) * batch_size] = batch_hash
    return result
