"""
Rollout buffer for on-policy PPO.

Stores trajectories of fixed length (update_interval steps), then computes
GAE-Lambda advantages and returns in-place before yielding mini-batches.
"""

import numpy as np
import torch
from typing import Generator, Tuple


class RolloutBuffer:
    """
    On-policy rollout buffer.

    Parameters
    ----------
    capacity : int
        Number of environment steps per PPO update (T in the paper).
    state_dim : int
    action_dim : int   (unused here, actions stored as scalars)
    gamma : float      discount factor γ
    lambda_gae : float GAE-λ parameter
    device : str
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.device = device
        self._ptr = 0
        self._full = False
        self._allocate()

    def _allocate(self) -> None:
        T, D = self.capacity, self.state_dim
        self.states     = np.zeros((T, D), dtype=np.float32)
        self.actions    = np.zeros(T, dtype=np.int64)
        self.log_probs  = np.zeros(T, dtype=np.float32)
        self.rewards    = np.zeros(T, dtype=np.float32)
        self.values     = np.zeros(T, dtype=np.float32)
        self.dones      = np.zeros(T, dtype=np.float32)
        self.advantages = np.zeros(T, dtype=np.float32)
        self.returns    = np.zeros(T, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        idx = self._ptr % self.capacity
        self.states[idx]    = state
        self.actions[idx]   = action
        self.log_probs[idx] = log_prob
        self.rewards[idx]   = reward
        self.values[idx]    = value
        self.dones[idx]     = float(done)
        self._ptr += 1
        if self._ptr >= self.capacity:
            self._full = True

    def is_ready(self) -> bool:
        return self._full

    def compute_gae(self, last_value: float) -> None:
        """
        Generalised Advantage Estimation (Schulman et al., 2015).

            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
        """
        gae = 0.0
        for t in reversed(range(self.capacity)):
            next_val = last_value if t == self.capacity - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + self.gamma * next_val * next_non_terminal
                - self.values[t]
            )
            gae = delta + self.gamma * self.lambda_gae * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def get_batches(
        self, batch_size: int
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """Yield shuffled mini-batches as torch tensors."""
        T = self.capacity
        # Normalise advantages
        adv = self.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        indices = np.random.permutation(T)
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            idx = indices[start:end]
            yield (
                torch.as_tensor(self.states[idx],    device=self.device),
                torch.as_tensor(self.actions[idx],   device=self.device),
                torch.as_tensor(self.log_probs[idx], device=self.device),
                torch.as_tensor(self.returns[idx],   device=self.device),
                torch.as_tensor(adv[idx],            device=self.device),
            )

    def reset(self) -> None:
        self._ptr = 0
        self._full = False
        self._allocate()
