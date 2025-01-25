import torch
from pilgrim import Trainer, Searcher, Pilgrim
from pilgrim.utils import load_cube_data, generate_inverse_moves

class ModelManager:
    def __init__(self, cube_size, cube_type, device_id=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", device_id)
        self.cube_size = cube_size
        self.cube_type = cube_type
        self.model = None
        self.trainer = None
        self.searcher = None

    def initialize_model(self, hd1, hd2, nrd, dropout_rate=0.1, activation_function="relu"):
        state_size = self._get_state_size()
        self.model = Pilgrim(
            state_size=state_size,
            hd1=hd1,
            hd2=hd2,
            nrd=nrd,
            dropout_rate=dropout_rate,
            activation_function=activation_function
        ).to(self.device)

    def _get_state_size(self):
        all_moves, _ = load_cube_data(self.cube_size, self.cube_type, self.device)
        return all_moves.size(1)

    def train_model(self, epochs, batch_size, lr, K_min, K_max, alpha):
        all_moves, move_names = load_cube_data(self.cube_size, self.cube_type, self.device)
        inverse_moves = torch.tensor(generate_inverse_moves(move_names), dtype=torch.int64, device=self.device)
        V0 = torch.arange(6, dtype=torch.int8, device=self.device).repeat_interleave(self._get_state_size() // 6)

        self.trainer = Trainer(
            net=self.model,
            num_epochs=epochs,
            device=self.device,
            batch_size=batch_size,
            lr=lr,
            name=f"cube{self.cube_size}_{self.cube_type}",
            K_min=K_min,
            K_max=K_max,
            all_moves=all_moves,
            inverse_moves=inverse_moves,
            V0=V0,
            α=alpha
        )
        self.trainer.run()

    def load_model(self, weights_path):
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def perform_beam_search(self, state, B, num_steps, num_attempts, verbose=0):
        all_moves, _ = load_cube_data(self.cube_size, self.cube_type, self.device)
        V0 = torch.arange(6, dtype=torch.int8, device=self.device).repeat_interleave(self._get_state_size() // 6)

        self.searcher = Searcher(
            model=self.model,
            all_moves=all_moves,
            V0=V0,
            device=self.device,
            verbose=verbose
        )
        return self.searcher.get_solution(state, B=B, num_steps=num_steps, num_attempts=num_attempts)