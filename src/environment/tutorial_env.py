"""
Gymnasium-compatible adaptive tutoring environment.

State space  (26-dim continuous):
    [0:10]  noisy per-topic knowledge estimates
    [10:20] one-hot current topic
    [20]    difficulty level
    [21]    engagement
    [22]    session progress
    [23]    consecutive correct / 10
    [24]    consecutive wrong / 10
    [25]    prerequisite readiness

Action space (11 discrete actions):
    0  EXPLAIN_CONCEPT
    1  SHOW_EXAMPLE
    2  ASK_EASY
    3  ASK_MEDIUM
    4  ASK_HARD
    5  GIVE_HINT
    6  INCREASE_DIFFICULTY
    7  DECREASE_DIFFICULTY
    8  SWITCH_TOPIC          (target chosen by orchestrator/bandit)
    9  REVIEW_PREVIOUS
    10 ENCOURAGE
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .student_simulator import (
    StudentSimulator, StudentProfile, TOPIC_PREREQUISITES
)
from .reward_function import RewardFunction, RewardConfig


ACTION_NAMES = [
    "EXPLAIN_CONCEPT",
    "SHOW_EXAMPLE",
    "ASK_EASY",
    "ASK_MEDIUM",
    "ASK_HARD",
    "GIVE_HINT",
    "INCREASE_DIFFICULTY",
    "DECREASE_DIFFICULTY",
    "SWITCH_TOPIC",
    "REVIEW_PREVIOUS",
    "ENCOURAGE",
]

STATE_DIM  = 26
ACTION_DIM = 11


class TutorialEnv:
    """
    Adaptive tutoring environment compatible with Gymnasium-style API.

    The orchestrator calls env.step(action, topic_override) where
    topic_override is only used when action == SWITCH_TOPIC (action 8).
    """

    def __init__(
        self,
        n_topics: int = 10,
        max_steps: int = 50,
        profile: Optional[StudentProfile] = None,
        reward_config: Optional[RewardConfig] = None,
        seed: int = 42,
    ):
        self.n_topics = n_topics
        self.max_steps = max_steps
        self.reward_config = reward_config or RewardConfig()
        self._rng = np.random.default_rng(seed)
        self._profile = profile  # None → randomise per episode

        self.state_dim  = STATE_DIM
        self.action_dim = ACTION_DIM

        self.student: Optional[StudentSimulator] = None
        self.reward_fn: Optional[RewardFunction] = None
        self._step_count = 0
        self._episode_knowledge_history: list = []
        self._episode_rewards: list = []

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self, profile: Optional[StudentProfile] = None) -> np.ndarray:
        profile = profile or self._profile or StudentProfile(
            self._rng.integers(0, len(StudentProfile))
        )
        self.student = StudentSimulator(
            n_topics=self.n_topics, profile=profile, rng=self._rng
        )
        self.reward_fn = RewardFunction(self.reward_config)
        self.reward_fn.reset()
        self._step_count = 0
        self._episode_knowledge_history = [self.student.true_knowledge.copy()]
        self._episode_rewards = []
        return self._get_obs()

    def step(
        self, action: int, topic_override: Optional[int] = None
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        assert self.student is not None, "Call reset() before step()"
        assert 0 <= action < self.action_dim

        self._step_count += 1

        # Prerequisite readiness before step (for reward shaping)
        prereqs = TOPIC_PREREQUISITES.get(self.student.current_topic, [])
        prereq_readiness = (
            float(np.mean([self.student.true_knowledge[p] for p in prereqs]))
            if prereqs else 1.0
        )

        answered, k_gain, eng_delta = self.student.step(action, new_topic=topic_override)

        reward = self.reward_fn.compute(
            action=action,
            answered_correctly=answered,
            knowledge_gain=k_gain,
            engagement=self.student.engagement,
            engagement_delta=eng_delta,
            current_topic=self.student.current_topic,
            true_knowledge=self.student.true_knowledge,
            prereq_knowledge=prereq_readiness,
        )

        done = (
            self._step_count >= self.max_steps
            or self.student.is_disengaged()
            or len(self.student.mastered_topics()) == self.n_topics
        )

        if done:
            reward += self.reward_fn.compute_terminal_bonus(self.student.true_knowledge)

        self._episode_knowledge_history.append(self.student.true_knowledge.copy())
        self._episode_rewards.append(reward)

        obs = self._get_obs()
        info = {
            "answered_correctly": answered,
            "knowledge_gain": k_gain,
            "engagement": self.student.engagement,
            "current_topic": self.student.current_topic,
            "mean_knowledge": float(np.mean(self.student.true_knowledge)),
            "n_mastered": len(self.student.mastered_topics()),
            "step": self._step_count,
        }
        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        progress = self._step_count / self.max_steps
        return self.student.get_observable_state(progress)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def select_next_topic(self) -> int:
        """
        Heuristic fallback for SWITCH_TOPIC when orchestrator has no bandit yet.
        Picks the unlocked topic with the lowest knowledge.
        """
        knowledge = self.student.true_knowledge
        best_topic, best_score = self.student.current_topic, 1.0
        for t in range(self.n_topics):
            prereqs = TOPIC_PREREQUISITES.get(t, [])
            prereqs_met = all(knowledge[p] >= 0.5 for p in prereqs)
            if prereqs_met and knowledge[t] < best_score:
                best_score = knowledge[t]
                best_topic = t
        return best_topic

    @property
    def current_topic(self) -> int:
        return self.student.current_topic if self.student else 0

    def episode_summary(self) -> Dict:
        history = np.array(self._episode_knowledge_history)
        return {
            "total_reward": sum(self._episode_rewards),
            "mean_knowledge_final": float(np.mean(history[-1])) if len(history) > 0 else 0.0,
            "knowledge_gain": float(np.mean(history[-1] - history[0])) if len(history) > 1 else 0.0,
            "n_mastered": len(self.student.mastered_topics()) if self.student else 0,
            "steps": self._step_count,
            "disengaged": self.student.is_disengaged() if self.student else False,
        }
