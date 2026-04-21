"""
Neural network architectures for PPO actor-critic and supporting modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def _init_weights(module: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal initialisation as recommended for RL."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


class MLP(nn.Module):
    """Feed-forward network with LayerNorm and optional output activation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        output_gain: float = 1.0,
    ):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
            in_d = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_d, output_dim)
        self.backbone.apply(lambda m: _init_weights(m, gain=1.0))
        _init_weights(self.head, gain=output_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class ActorCritic(nn.Module):
    """
    Shared-backbone actor-critic for PPO.

    Architecture:
        shared backbone  → 2-layer MLP with LayerNorm + Tanh
        actor head       → softmax over discrete actions
        critic head      → scalar state-value estimate
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        # actor: small gain → near-uniform initial policy
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        # critic: larger gain → better initial value estimates
        self.critic_head = nn.Linear(hidden_dim, 1)

        self.backbone.apply(lambda m: _init_weights(m, gain=1.0))
        _init_weights(self.actor_head, gain=0.01)
        _init_weights(self.critic_head, gain=1.0)

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_probs, state_value)."""
        features = self.backbone(state)
        action_probs = F.softmax(self.actor_head(features), dim=-1)
        value = self.critic_head(features)
        return action_probs, value

    def get_action_and_value(
        self, state: torch.Tensor, action: torch.Tensor = None
    ) -> tuple:
        """
        Sample an action (or evaluate given action).
        Returns (action, log_prob, entropy, value).
        """
        action_probs, value = self(state)
        dist = Categorical(action_probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> tuple[int, float, float]:
        """Greedy inference. Returns (action_int, log_prob_float, value_float)."""
        action, log_prob, _, value = self.get_action_and_value(state)
        return action.item(), log_prob.item(), value.item()
