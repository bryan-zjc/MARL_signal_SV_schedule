import torch
import numpy as np
from typing import Dict


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",):

        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        # self._alphas = torch.zeros((buffer_size, 8), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)
    
    def _to_tensor_int64(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.int64, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["states"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        print('----- load offline data with states shape', data["states"].shape)
        self._states[:n_transitions] = self._to_tensor(data["states"])
        self._actions[:n_transitions] = self._to_tensor_int64(data["actions"][..., None])
        # self._alphas[:n_transitions] = self._to_tensor_int64(data["alphas"][..., None])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_states"])
        self._dones[:n_transitions] = self._to_tensor(data["dones"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        # alphas = self._alphas[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        # return [states, actions, alphas, rewards, next_states, dones]
        return [states, actions, rewards, next_states, dones]

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        # alpha: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        # self._alphas[self._pointer] = self._to_tensor(alpha)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)
        
        

