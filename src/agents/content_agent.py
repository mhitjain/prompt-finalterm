"""
Content Agent — uses Linear Thompson Sampling to select the best content
presentation style for the current student context.

Arms (5 content styles):
  0 → EXPLAIN_CONCEPT   (textual conceptual explanation)
  1 → SHOW_EXAMPLE      (worked example)
  2 → VISUAL_DIAGRAM    (maps to SHOW_EXAMPLE with visual hint)
  3 → REVIEW_PREVIOUS   (spaced repetition review)
  4 → GIVE_HINT         (Socratic hint/scaffolding)

The agent personalises content delivery by learning which style produces
the highest engagement-weighted knowledge gain for each learner type.
"""

import numpy as np
from typing import Any, Dict, Optional

from .base_agent import BaseAgent
from ..rl.contextual_bandits import LinThompsonSampling


CONTEXT_DIM  = 14
N_ARMS       = 5
CONTENT_NAMES = [
    "Explain Concept",
    "Show Example",
    "Visual/Diagram",
    "Review Previous",
    "Give Hint",
]
# Map content arm → tutorial env action
ARM_TO_ACTION = {0: 0, 1: 1, 2: 1, 3: 9, 4: 5}


def _extract_content_context(obs: np.ndarray) -> np.ndarray:
    """Extract 14-dim context for content style selection."""
    knowledge   = obs[0:10].copy()
    engagement  = obs[21:22]
    difficulty  = obs[20:21]
    consec_corr = obs[23:24]
    prereq      = obs[25:26]
    return np.concatenate([knowledge, engagement, difficulty, consec_corr, prereq])


class ContentAgent(BaseAgent):
    """
    Contextual bandit agent for content presentation style selection.

    Learns to match content style to student profile and current knowledge
    state to maximise engagement-weighted learning.
    """

    def __init__(
        self,
        agent_id: str = "content_agent",
        alpha: float = 1.0,
        tools=None,
    ):
        super().__init__(agent_id, tools)
        self.bandit = LinThompsonSampling(
            n_arms=N_ARMS, context_dim=CONTEXT_DIM, alpha=alpha
        )
        self._last_arm: Optional[int] = None
        self._last_context: Optional[np.ndarray] = None

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
        Returns env action for content delivery.
        """
        self._step_count += 1
        ctx = _extract_content_context(observation)
        arm = self.bandit.select_arm(ctx)
        self._last_arm = arm
        self._last_context = ctx.copy()
        return ARM_TO_ACTION[arm]

    # ------------------------------------------------------------------
    # Bandit update
    # ------------------------------------------------------------------

    def step_done(self, feedback: Dict[str, Any]) -> None:
        """Update posterior with engagement-weighted knowledge gain."""
        if self._last_arm is None:
            return
        k_gain = max(feedback.get("knowledge_gain", 0.0), 0.0)
        engagement = feedback.get("engagement", 0.5)
        # Combined reward: learning × engagement quality
        reward = k_gain * 3.0 + engagement * 0.2
        self.bandit.update(self._last_arm, self._last_context, reward)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        base = super().get_stats()
        return {
            **base,
            "arm_stats": {
                CONTENT_NAMES[a]: {
                    "n_pulls":    self.bandit.stats[a].n_pulls,
                    "mean_reward": self.bandit.stats[a].mean_reward,
                }
                for a in range(N_ARMS)
            },
        }
