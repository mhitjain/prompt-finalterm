"""
Assessment Agent — uses Linear Thompson Sampling to select the optimal
question difficulty for each student context.

The bandit learns which of three assessment modes (easy/medium/hard) maximises
immediate learning gain for each student context, complementing the PPO policy's
higher-level strategy.

Arms:
  0 → easy     (IRT difficulty 0.25)
  1 → medium   (IRT difficulty 0.50)
  2 → hard     (IRT difficulty 0.80)

Context features (14-dim subset of full state):
  [0:10]  knowledge per topic
  [10]    engagement
  [11]    difficulty_level
  [12]    recent accuracy proxy (consecutive_correct / 10)
  [13]    prerequisite readiness
"""

import numpy as np
from typing import Any, Dict, Optional

from .base_agent import BaseAgent
from ..rl.contextual_bandits import LinThompsonSampling, BanditStats


CONTEXT_DIM = 14
N_ARMS      = 3
ARM_NAMES   = ["easy", "medium", "hard"]
ARM_TO_ENV_ACTION = {0: 2, 1: 3, 2: 4}   # bandit arm → tutorial env action


def _extract_assessment_context(obs: np.ndarray, current_topic: int) -> np.ndarray:
    """Extract the 14-dim context relevant for question difficulty selection."""
    knowledge   = obs[0:10].copy()
    engagement  = obs[21:22]
    difficulty  = obs[20:21]
    consec_corr = obs[23:24]
    prereq_read = obs[25:26]
    return np.concatenate([knowledge, engagement, difficulty, consec_corr, prereq_read])


class AssessmentAgent(BaseAgent):
    """
    Contextual bandit agent for adaptive question difficulty selection.

    At each step, it receives the student context and decides which
    question difficulty maximises expected learning reward.
    """

    def __init__(
        self,
        agent_id: str = "assessment_agent",
        alpha: float = 1.0,
        use_ucb_warmup: int = 20,
        tools=None,
    ):
        super().__init__(agent_id, tools)
        self.bandit = LinThompsonSampling(
            n_arms=N_ARMS, context_dim=CONTEXT_DIM, alpha=alpha
        )
        self.use_ucb_warmup = use_ucb_warmup
        self._last_arm: Optional[int] = None
        self._last_context: Optional[np.ndarray] = None
        self._rewards_per_arm = [[] for _ in range(N_ARMS)]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._last_arm = None
        self._last_context = None

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, observation: np.ndarray, context: Dict[str, Any]) -> int:
        """
        Select a question difficulty arm and return the corresponding env action.

        Parameters
        ----------
        observation : full 26-dim student state
        context     : shared context (contains current_topic)

        Returns
        -------
        env action index (2=ASK_EASY, 3=ASK_MEDIUM, 4=ASK_HARD)
        """
        self._step_count += 1
        current_topic = context.get("current_topic", 0)
        ctx = _extract_assessment_context(observation, current_topic)

        # Use LinUCB during warm-up (ensures each arm is explored sufficiently)
        total_pulls = sum(s.n_pulls for s in self.bandit.stats)
        if total_pulls < self.use_ucb_warmup:
            arm = self.bandit.select_arm_ucb(ctx)
        else:
            arm = self.bandit.select_arm(ctx)

        self._last_arm = arm
        self._last_context = ctx.copy()
        return ARM_TO_ENV_ACTION[arm]

    # ------------------------------------------------------------------
    # Bandit update
    # ------------------------------------------------------------------

    def step_done(self, feedback: Dict[str, Any]) -> None:
        """Update bandit posterior with observed reward for last selected arm."""
        if self._last_arm is None or self._last_context is None:
            return
        # Reward = knowledge_gain + 0.5 * correct + 0.1 * engagement
        reward = (
            feedback.get("knowledge_gain", 0.0) * 3.0
            + float(feedback.get("answered_correctly", False)) * 0.5
            + feedback.get("engagement", 0.5) * 0.1
        )
        self.bandit.update(self._last_arm, self._last_context, reward)
        self._rewards_per_arm[self._last_arm].append(reward)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        base = super().get_stats()
        arm_stats = self.bandit.arm_summary()
        return {
            **base,
            "arm_stats": {
                ARM_NAMES[a]: arm_stats[a] for a in range(N_ARMS)
            },
            "preferred_arm": ARM_NAMES[
                max(range(N_ARMS), key=lambda a: self.bandit.stats[a].mean_reward)
            ] if any(s.n_pulls > 0 for s in self.bandit.stats) else "unknown",
        }
