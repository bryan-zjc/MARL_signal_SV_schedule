import torch
import numpy as np
from typing import Dict


class ReplayBuffer:
    def __init__(
        self,
        RV_state_dim: int,
        PL_state_dim: int,
        seq_len: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",):

        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._RV_states = torch.zeros((buffer_size, seq_len, RV_state_dim), dtype=torch.float32, device=device)
        self._PL_states = torch.zeros((buffer_size, PL_state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        # self._alphas = torch.zeros((buffer_size, 8), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_RV_states = torch.zeros((buffer_size, seq_len, RV_state_dim), dtype=torch.float32, device=device)
        self._next_PL_states = torch.zeros((buffer_size, PL_state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)
    
    def _to_tensor_int64(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.int64, device=self._device)


    def sample(self, batch_size: int):
        indices = np.random.randint(0, self._size, size=batch_size)
        RV_states = self._RV_states[indices]
        PL_states = self._PL_states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_RV_states = self._next_RV_states[indices]
        next_PL_states = self._next_PL_states[indices]
        dones = self._dones[indices]
        # return [states, actions, alphas, rewards, next_states, dones]
        return [RV_states, PL_states, actions, rewards, next_RV_states, next_PL_states, dones]

    def add_transition(
        self,
        RV_state: torch.tensor,
        PL_state: np.ndarray,
        action: np.ndarray,
        # alpha: np.ndarray,
        reward: float,
        next_RV_state: torch.tensor,
        next_PL_state: np.ndarray,
        done: bool,):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._RV_states[self._pointer] = self._to_tensor(RV_state)
        self._PL_states[self._pointer] = self._to_tensor(PL_state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_RV_states[self._pointer] = self._to_tensor(next_RV_state)
        self._next_PL_states[self._pointer] = self._to_tensor(next_PL_state)
        self._dones[self._pointer] = self._to_tensor(done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)
        
        

