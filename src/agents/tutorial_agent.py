"""
Tutorial Agent — the primary RL agent trained with PPO.

Responsibilities:
  - Selects high-level teaching actions (explain, ask question, hint, etc.)
  - Maintains the PPO rollout buffer and triggers policy updates
  - Reports learning metrics to the orchestrator
"""

import numpy as np
from typing import Any, Dict, Optional

from .base_agent import BaseAgent
from ..rl.ppo import PPO, PPOConfig
from ..environment.tutorial_env import STATE_DIM, ACTION_DIM, ACTION_NAMES


class TutorialAgent(BaseAgent):
    """
    PPO-based tutorial agent.

    The agent observes the full 26-dim student state and selects one of
    11 discrete teaching actions per step.
    """

    def __init__(
        self,
        agent_id: str = "tutorial_agent",
        ppo_config: Optional[PPOConfig] = None,
        device: str = "cpu",
        tools=None,
    ):
        super().__init__(agent_id, tools)
        self.ppo = PPO(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            config=ppo_config or PPOConfig(),
            device=device,
        )
        self._last_obs: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None
        self._last_log_prob: Optional[float] = None
        self._last_value: Optional[float] = None
        self._total_timesteps = 0
        self._update_count = 0
        self._episode_rewards: list = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._last_obs = None
        self._last_action = None
        self._last_log_prob = None
        self._last_value = None
        self._episode_rewards = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, observation: np.ndarray, context: Dict[str, Any]) -> int:
        """
        Select a teaching action using the current PPO policy.

        Parameters
        ----------
        observation : 26-dim state vector from TutorialEnv
        context     : shared context dict from orchestrator (may override certain actions)

        Returns
        -------
        action index (int 0–10)
        """
        self._step_count += 1
        self._total_timesteps += 1

        action, log_prob, value = self.ppo.select_action(observation)

        # Mask illegal actions based on context (e.g., don't SWITCH_TOPIC if already optimal)
        action = self._apply_action_mask(action, observation, context)

        self._last_obs = observation
        self._last_action = action
        self._last_log_prob = log_prob
        self._last_value = value
        return action

    def _apply_action_mask(
        self, action: int, obs: np.ndarray, context: Dict
    ) -> int:
        """
        Override PPO's action if it would be clearly harmful.
        E.g., don't ask HARD question when student knowledge < 0.2.
        """
        knowledge_current_topic = float(obs[context.get("current_topic", 0)])
        engagement = float(obs[21])

        # If engagement is critically low, prioritise encouragement
        if engagement < 0.20 and action != 10:
            return 10  # ENCOURAGE

        # Don't ask hard question if student knowledge very low
        if action == 4 and knowledge_current_topic < 0.15:
            return 2  # downgrade to ASK_EASY

        # Don't explain if topic already mastered
        if action == 0 and knowledge_current_topic > 0.85:
            return 3  # ask medium instead

        return action

    # ------------------------------------------------------------------
    # PPO buffer management
    # ------------------------------------------------------------------

    def store_transition(self, reward: float, done: bool) -> None:
        """Store the last (s, a, log_prob, r, v, done) into the PPO buffer."""
        if self._last_obs is None:
            return
        self.ppo.buffer.add(
            state=self._last_obs,
            action=self._last_action,
            log_prob=self._last_log_prob,
            reward=reward,
            value=self._last_value,
            done=done,
        )
        self._episode_rewards.append(reward)

    def maybe_update(self, last_obs: np.ndarray) -> Optional[Dict]:
        """Trigger a PPO update if the buffer is full."""
        if self.ppo.buffer.is_ready():
            metrics = self.ppo.update(last_obs)
            self._update_count += 1
            self.send(
                receiver="orchestrator",
                msg_type="training_metrics",
                payload={**metrics, "timestep": self._total_timesteps},
            )
            return metrics
        return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        return {
            **super().get_stats(),
            "total_timesteps": self._total_timesteps,
            "ppo_updates": self._update_count,
            "last_action_name": (
                ACTION_NAMES[self._last_action]
                if self._last_action is not None else "N/A"
            ),
            "episode_mean_reward": (
                float(np.mean(self._episode_rewards)) if self._episode_rewards else 0.0
            ),
        }

    def save(self, path: str) -> None:
        self.ppo.save(path)

    def load(self, path: str) -> None:
        self.ppo.load(path)
