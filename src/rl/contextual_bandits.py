"""
Linear Thompson Sampling for Contextual Bandits.

Theory
------
For each arm a we maintain a Bayesian linear regression model:
    E[r | x, a] = x^T θ_a    where x is the context vector

Prior:   θ_a ~ N(0, α^{-1} I)
Posterior after n observations:
    B_a  = α I + X_a^T X_a              (precision matrix)
    μ_a  = B_a^{-1} (α * 0 + X_a^T y_a) = B_a^{-1} f_a

Sampling:
    θ̃_a ~ N(μ_a, B_a^{-1})
    Select arm  argmax_a  x^T θ̃_a

Update (Sherman-Morrison rank-1 update for efficiency):
    B_a  ← B_a + x x^T
    f_a  ← f_a + r * x
    μ_a  ← B_a^{-1} f_a

Reference: Agrawal & Goyal (2013), "Thompson Sampling for Contextual Bandits
           with Linear Payoffs."
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BanditStats:
    n_pulls: int = 0
    total_reward: float = 0.0
    reward_history: List[float] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(1, self.n_pulls)


class LinThompsonSampling:
    """
    Linear Thompson Sampling contextual bandit for n_arms arms.

    Parameters
    ----------
    n_arms      : number of arms (e.g., content styles, question types)
    context_dim : dimension of context vector
    alpha       : prior precision (higher → less exploration initially)
    """

    def __init__(self, n_arms: int, context_dim: int, alpha: float = 1.0):
        self.n_arms      = n_arms
        self.context_dim = context_dim
        self.alpha       = alpha

        # Per-arm Bayesian linear regression state
        self.B: List[np.ndarray] = [
            np.eye(context_dim) * alpha for _ in range(n_arms)
        ]
        self.f: List[np.ndarray] = [
            np.zeros(context_dim) for _ in range(n_arms)
        ]
        self.mu: List[np.ndarray] = [
            np.zeros(context_dim) for _ in range(n_arms)
        ]
        # Cholesky factors for efficient sampling (updated lazily)
        self._B_inv: List[Optional[np.ndarray]] = [None] * n_arms
        self._dirty: List[bool] = [True] * n_arms

        self.stats: List[BanditStats] = [BanditStats() for _ in range(n_arms)]
        self._t = 0   # global time step

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select_arm(self, context: np.ndarray) -> int:
        """
        Sample θ̃_a from posterior for each arm, return argmax_a x^T θ̃_a.
        """
        context = np.asarray(context, dtype=np.float64)
        assert context.shape == (self.context_dim,), (
            f"Expected context shape ({self.context_dim},), got {context.shape}"
        )

        expected_rewards = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            B_inv = self._get_B_inv(a)
            # Sample parameter vector from posterior
            try:
                theta_sample = np.random.multivariate_normal(
                    self.mu[a], self.alpha * B_inv
                )
            except np.linalg.LinAlgError:
                # Fallback: use posterior mean (greedy)
                theta_sample = self.mu[a]
            expected_rewards[a] = context @ theta_sample

        selected = int(np.argmax(expected_rewards))
        self._t += 1
        return selected

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """
        Update the posterior for the selected arm with the observed reward.
        """
        context = np.asarray(context, dtype=np.float64)
        # Rank-1 update of precision matrix
        self.B[arm] += np.outer(context, context)
        self.f[arm] += reward * context
        self._dirty[arm] = True   # invalidate cached inverse

        # Recompute posterior mean
        B_inv = self._get_B_inv(arm)
        self.mu[arm] = B_inv @ self.f[arm]

        # Record stats
        self.stats[arm].n_pulls += 1
        self.stats[arm].total_reward += reward
        self.stats[arm].reward_history.append(reward)

    # ------------------------------------------------------------------
    # UCB fallback for cold-start (first few steps per arm)
    # ------------------------------------------------------------------

    def select_arm_ucb(self, context: np.ndarray, delta: float = 0.1) -> int:
        """
        LinUCB selection as warm-start alternative to Thompson Sampling.
        Used when fewer than context_dim pulls have been made for any arm.
        """
        context = np.asarray(context, dtype=np.float64)
        ucb_scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            B_inv = self._get_B_inv(a)
            exploitation = self.mu[a] @ context
            # Confidence width (LinUCB formula)
            width = np.sqrt(context @ B_inv @ context)
            alpha_ucb = np.sqrt(np.log(max(1, self.stats[a].n_pulls + 1) / delta))
            ucb_scores[a] = exploitation + alpha_ucb * width
        return int(np.argmax(ucb_scores))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_B_inv(self, arm: int) -> np.ndarray:
        if self._dirty[arm] or self._B_inv[arm] is None:
            try:
                self._B_inv[arm] = np.linalg.inv(self.B[arm])
            except np.linalg.LinAlgError:
                self._B_inv[arm] = np.linalg.pinv(self.B[arm])
            self._dirty[arm] = False
        return self._B_inv[arm]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def arm_summary(self) -> Dict[int, Dict]:
        return {
            a: {
                "n_pulls":    self.stats[a].n_pulls,
                "mean_reward": self.stats[a].mean_reward,
                "theta_norm": float(np.linalg.norm(self.mu[a])),
            }
            for a in range(self.n_arms)
        }

    def reset(self) -> None:
        self.__init__(self.n_arms, self.context_dim, self.alpha)


# ---------------------------------------------------------------------------
# Convenience: multi-armed epsilon-greedy baseline (for ablation comparison)
# ---------------------------------------------------------------------------

class EpsilonGreedyBandit:
    """Simple ε-greedy bandit for ablation studies."""

    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self, context: np.ndarray = None) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
