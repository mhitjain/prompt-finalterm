"""
Reward function engineering for the adaptive tutoring environment.

Design principles:
  - Dense rewards guide learning at every step
  - Sparse mastery bonuses incentivise long-horizon planning
  - Engagement penalty prevents reward hacking via disengagement
  - Curriculum shaping bonus encourages prerequisite ordering
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RewardConfig:
    knowledge_gain_scale: float = 5.0      # multiplier for per-step knowledge gain
    correct_answer_bonus: float = 0.5      # bonus for correct response
    wrong_answer_penalty: float = -0.3     # penalty for wrong answer
    mastery_bonus: float = 10.0            # sparse reward when topic mastered
    disengagement_penalty: float = -5.0    # triggered if engagement < threshold
    engagement_reward_scale: float = 0.3   # reward for maintaining high engagement
    time_penalty: float = -0.01            # per-step cost (encourages efficiency)
    curriculum_bonus: float = 1.0          # reward for teaching in prereq order
    engagement_threshold: float = 0.15    # below this → disengagement penalty
    mastery_threshold: float = 0.85


class RewardFunction:
    """Computes shaped rewards from environment transitions."""

    def __init__(self, config: RewardConfig = None):
        self.cfg = config or RewardConfig()
        self._previously_mastered: set = set()

    def reset(self) -> None:
        self._previously_mastered = set()

    def compute(
        self,
        action: int,
        answered_correctly: bool,
        knowledge_gain: float,
        engagement: float,
        engagement_delta: float,
        current_topic: int,
        true_knowledge: np.ndarray,
        prereq_knowledge: float,
    ) -> float:
        cfg = self.cfg
        reward = cfg.time_penalty  # always apply time cost

        # --- knowledge gain reward ---
        reward += cfg.knowledge_gain_scale * max(knowledge_gain, 0.0)

        # --- answer correctness ---
        if action in (2, 3, 4):  # question actions
            if answered_correctly:
                reward += cfg.correct_answer_bonus
            else:
                reward += cfg.wrong_answer_penalty

        # --- engagement ---
        if engagement < cfg.engagement_threshold:
            reward += cfg.disengagement_penalty
        else:
            reward += cfg.engagement_reward_scale * engagement

        # --- mastery bonus (sparse) ---
        mastered = true_knowledge[current_topic] >= cfg.mastery_threshold
        if mastered and current_topic not in self._previously_mastered:
            self._previously_mastered.add(current_topic)
            reward += cfg.mastery_bonus

        # --- curriculum shaping: bonus if prerequisites well-learned ---
        if prereq_knowledge > 0.7 and knowledge_gain > 0:
            reward += cfg.curriculum_bonus * prereq_knowledge * knowledge_gain

        return float(reward)

    def compute_terminal_bonus(self, true_knowledge: np.ndarray) -> float:
        """Episode-end bonus proportional to overall knowledge achieved."""
        return float(np.sum(true_knowledge) * 2.0)
