import torch
import json
import os

def generate_permutation_data(permutation, permutation_size, permutation_type):
    # Define elements as [0, 1, 2, 3, 4, 5]
    if permutation == 'standard':
        elements = list(range(permutation_size))
        permutations = []
        names = []

        # Generate transpositions (2-cycles)
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                if i == 0 and j == 1:
                    # Transposition (i <-> j)
                    transposition = elements[:]
                    transposition[i], transposition[j] = transposition[j], transposition[i]
                    permutations.append(transposition)
                    names.append(f"t{i}{j}")  # Name for transposition

                    # Inverse of transposition is itself
                    permutations.append(transposition[:])
                    names.append(f"t{i}{j}'")
        
        # Generate a full cycle (0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0)
        full_cycle = elements[:]
        full_cycle_rotated = full_cycle[-1:] + full_cycle[:-1]  # Rotate right by 1
        print(full_cycle_rotated)
        permutations.append(full_cycle_rotated)
        names.append("full_cycle")

        # Generate inverse full cycle (5 -> 4 -> 3 -> 2 -> 1 -> 0 -> 5)
        inverse_full_cycle = elements[:]
        inverse_full_cycle = inverse_full_cycle[1:] + inverse_full_cycle[:1]  # Rotate left by 1
        permutations.append(inverse_full_cycle)
        names.append("full_cycle'")

        # Combine data into a JSON-compatible structure
        standard_data = {"actions": permutations, "names": names}

        # Save to JSON
        output_path = f"generators/{permutation_type}_{permutation}{permutation_size}.json"
        with open(output_path, "w") as f:
            json.dump(standard_data, f)

        print(f"File saved to {output_path}")
    else:
        raise ValueError(f"Invalid permutation: {permutation}")

    return standard_data

def generate_permutation_data_test(permutation, permutation_size, permutation_type):
    # Define elements as [0, 1, 2, 3, 4, 5]
    if permutation == 'standard':
        elements = list(range(permutation_size))
        permutations = []
        names = []

        # Generate transpositions (2-cycles)
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                if i == 0 and j == 1:
                    # Transposition (i <-> j)
                    transposition = elements[:]
                    transposition[i], transposition[j] = transposition[j], transposition[i]
                    permutations.append(transposition)
                    names.append(f"t{i}{j}")  # Name for transposition

                    # # Inverse of transposition is itself
                    # permutations.append(transposition[:])
                    # names.append(f"t{i}{j}'")
        
        # Generate a full cycle (0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0)
        full_cycle = elements[:]
        full_cycle_rotated = full_cycle[-1:] + full_cycle[:-1]  # Rotate right by 1
        # print(full_cycle_rotated)
        permutations.append(full_cycle_rotated)
        names.append("full_cycle")

        # # Generate inverse full cycle (5 -> 4 -> 3 -> 2 -> 1 -> 0 -> 5)
        # inverse_full_cycle = elements[:]
        # inverse_full_cycle = inverse_full_cycle[1:] + inverse_full_cycle[:1]  # Rotate left by 1
        # permutations.append(inverse_full_cycle)
        # names.append("full_cycle'")

        # Combine data into a JSON-compatible structure
        standard_data = {"actions": permutations, "names": names}

        # Save to JSON
        output_path = f"generators/{permutation_type}_{permutation}{permutation_size}.json"
        with open(output_path, "w") as f:
            json.dump(standard_data, f)

        # print(f"File saved to {output_path}")
    else:
        raise ValueError(f"Invalid permutation: {permutation}")

    return standard_data

def load_permutation_data(permutation, permutation_size, permutation_type, device):
    """Load permutation data based on permutation size and type (qtm or all)."""
    file_path = f"generators/{permutation_type}_{permutation}{permutation_size}.json"
    
    generate_permutation_data(permutation, permutation_size, permutation_type)
    with open(file_path, 'rb') as f:
        data = json.load(f)

    actions = data["actions"]
    action_names = data["names"]
    
    return torch.tensor(actions, dtype=torch.int64, device=device), action_names

def load_permutation_data_test(permutation, permutation_size, permutation_type, device):
    """Load permutation data based on permutation size and type (qtm or all)."""
    file_path = f"generators/{permutation_type}_{permutation}{permutation_size}.json"
    
    generate_permutation_data_test(permutation, permutation_size, permutation_type)
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

def state2hash(states, hash_vec, batch_size=2**14):
    """Convert states to hashes."""
    num_batches = (states.size(0) + batch_size - 1) // batch_size
    result = torch.empty(states.size(0), dtype=torch.int64, device=states.device)
    
    for i in range(num_batches):
        batch = states[i * batch_size:(i + 1) * batch_size].to(torch.int64)
        batch_hash = torch.sum(hash_vec * batch, dim=1)
        result[i * batch_size:(i + 1) * batch_size] = batch_hash
    return result

# def get_unique_states(states, states_bad_hashed, hash_vec, batch_size=2**14):
#     """Filter unique states by removing duplicates based on hash."""
#     idx1 = torch.arange(states.size(0), dtype=torch.int64, device=states.device)
#     hashed = state2hash(states, hash_vec, batch_size)
#     mask1  = ~torch.isin(hashed, states_bad_hashed)
#     hashed = hashed[mask1]
#     hashed_sorted, idx2 = torch.sort(hashed)
#     mask2 = torch.concat((torch.tensor([True], device=states.device), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
#     return states[mask1][idx2[mask2]], idx1[mask1][idx2[mask2]] 

# def get_unique_hashed_states_idx(states, states_bad_hashed):
#     """Filter unique hashed states by removing duplicates"""
#     mask1  = ~torch.isin(hashed, states_bad_hashed)
#     hashed = hashed[mask1]
#     hashed_sorted, idx2 = torch.sort(hashed)
#     mask2 = torch.concat((torch.tensor([True], device=states.device), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
#     return idx1[mask1][idx2[mask2]] 

