"""
Proximal Policy Optimisation (Schulman et al., 2017).

PPO clips the surrogate objective to prevent destructively large updates:

    L^CLIP(θ) = E_t [ min(r_t(θ) Â_t,  clip(r_t(θ), 1-ε, 1+ε) Â_t) ]

where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the probability ratio.

Total loss:
    L(θ) = -L^CLIP + c1 * L^VF - c2 * S[π_θ](s_t)

with value coefficient c1, entropy bonus coefficient c2.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Dict, List, Optional

from .networks import ActorCritic
from .buffer import RolloutBuffer


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip_epsilon: float = 0.2
    epochs: int = 4
    batch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_interval: int = 2048
    hidden_dim: int = 256


class PPO:
    """
    PPO implementation for the tutorial agent.

    Usage
    -----
    ppo = PPO(state_dim=26, action_dim=11)
    obs = env.reset()
    for _ in range(total_timesteps):
        action, log_prob, value = ppo.select_action(obs)
        next_obs, reward, done, _ = env.step(action)
        ppo.buffer.add(obs, action, log_prob, reward, value, done)
        if ppo.buffer.is_ready():
            ppo.update(last_value)
        obs = env.reset() if done else next_obs
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: PPOConfig = None,
        device: str = "cpu",
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.cfg = config or PPOConfig()
        self.device = device

        self.policy = ActorCritic(
            state_dim, action_dim, hidden_dim=self.cfg.hidden_dim
        ).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.lr, eps=1e-5)

        # Linear LR anneal (optional, improves stability)
        self._total_updates = 0

        self.buffer = RolloutBuffer(
            capacity=self.cfg.update_interval,
            state_dim=state_dim,
            gamma=self.cfg.gamma,
            lambda_gae=self.cfg.lambda_gae,
            device=device,
        )

        # Metrics tracked across updates
        self.metrics_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> tuple[int, float, float]:
        """Sample action from current policy. Returns (action, log_prob, value)."""
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob, _, value = self.policy.get_action_and_value(state)
        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, value = self.policy(state)
        return value.item()

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, last_obs: np.ndarray) -> Dict[str, float]:
        """
        Run K epochs of PPO updates on the current rollout buffer.
        Returns dict of training metrics.
        """
        last_value = self.get_value(last_obs)
        self.buffer.compute_gae(last_value)

        policy_losses, value_losses, entropy_losses, clip_fracs = [], [], [], []

        for _ in range(self.cfg.epochs):
            for states, actions, old_log_probs, returns, advantages in self.buffer.get_batches(
                self.cfg.batch_size
            ):
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    states, actions
                )

                # PPO clipped surrogate objective
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                # Clip fraction (diagnostic: fraction of ratios clipped)
                clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_epsilon).float().mean()
                clip_fracs.append(clip_frac.item())

                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (clipped for stability)
                value_loss = 0.5 * ((new_values - returns) ** 2).mean()

                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()

                total_loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    + self.cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(-entropy_loss.item())

        self.buffer.reset()
        self._total_updates += 1

        metrics = {
            "policy_loss":   float(np.mean(policy_losses)),
            "value_loss":    float(np.mean(value_losses)),
            "entropy":       float(np.mean(entropy_losses)),
            "clip_fraction": float(np.mean(clip_fracs)),
            "update":        self._total_updates,
        }
        self.metrics_history.append(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({"policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
