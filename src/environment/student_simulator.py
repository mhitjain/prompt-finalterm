"""
Student simulator using Item Response Theory (IRT) and Bayesian Knowledge Tracing.

IRT models the probability of a correct answer as:
    P(correct | θ, b, a) = 1 / (1 + exp(-a * (θ - b)))
where θ = student ability, b = item difficulty, a = item discrimination.

Knowledge update follows a simplified BKT model:
    P(known | correct) increases by learning_rate * (1 - current_knowledge)
    P(known | wrong)   decreases by forgetting_rate * current_knowledge
"""

import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Tuple


class StudentProfile(IntEnum):
    FAST_LEARNER = 0
    SLOW_LEARNER = 1
    VISUAL_LEARNER = 2
    PRACTICE_LEARNER = 3


# Topic names for human readability
TOPIC_NAMES = [
    "Basic Arithmetic",     # 0
    "Algebra Basics",       # 1
    "Geometry Basics",      # 2
    "Statistics Basics",    # 3
    "Linear Algebra",       # 4
    "Calculus Intro",       # 5
    "Probability",          # 6
    "Advanced Statistics",  # 7
    "ML Basics",            # 8
    "Deep Learning",        # 9
]

# Prerequisite DAG: topic -> list of required prerequisites
TOPIC_PREREQUISITES = {
    0: [],
    1: [0],
    2: [0],
    3: [0],
    4: [1],
    5: [1],
    6: [3],
    7: [6],
    8: [4, 5, 6],
    9: [8],
}


@dataclass
class ProfileParams:
    """Behavioural parameters for each learner archetype."""
    learning_rate: float          # base knowledge gain per correct interaction
    forgetting_rate: float        # knowledge decay per incorrect interaction
    engagement_decay: float       # per-step engagement loss
    engagement_recovery: float    # engagement gain after success
    action_multipliers: Dict[int, float] = field(default_factory=dict)  # per-action learning boost


# Action indices (must match TutorialEnv.ACTIONS)
ACTION_EXPLAIN    = 0
ACTION_EXAMPLE    = 1
ACTION_EASY       = 2
ACTION_MEDIUM     = 3
ACTION_HARD       = 4
ACTION_HINT       = 5
ACTION_INC_DIFF   = 6
ACTION_DEC_DIFF   = 7
ACTION_SWITCH     = 8
ACTION_REVIEW     = 9
ACTION_ENCOURAGE  = 10

PROFILE_PARAMS: Dict[StudentProfile, ProfileParams] = {
    StudentProfile.FAST_LEARNER: ProfileParams(
        learning_rate=0.15,
        forgetting_rate=0.02,
        engagement_decay=0.005,
        engagement_recovery=0.08,
        action_multipliers={
            ACTION_HARD: 1.6,
            ACTION_MEDIUM: 1.2,
            ACTION_EASY: 0.7,
            ACTION_EXPLAIN: 0.9,
            ACTION_EXAMPLE: 1.0,
        },
    ),
    StudentProfile.SLOW_LEARNER: ProfileParams(
        learning_rate=0.05,
        forgetting_rate=0.06,
        engagement_decay=0.012,
        engagement_recovery=0.05,
        action_multipliers={
            ACTION_EASY: 1.5,
            ACTION_HINT: 1.4,
            ACTION_EXPLAIN: 1.3,
            ACTION_MEDIUM: 0.9,
            ACTION_HARD: 0.5,
        },
    ),
    StudentProfile.VISUAL_LEARNER: ProfileParams(
        learning_rate=0.10,
        forgetting_rate=0.04,
        engagement_decay=0.007,
        engagement_recovery=0.07,
        action_multipliers={
            ACTION_EXAMPLE: 1.8,
            ACTION_EXPLAIN: 1.4,
            ACTION_REVIEW: 1.2,
            ACTION_EASY: 1.0,
            ACTION_HARD: 0.8,
        },
    ),
    StudentProfile.PRACTICE_LEARNER: ProfileParams(
        learning_rate=0.12,
        forgetting_rate=0.03,
        engagement_decay=0.006,
        engagement_recovery=0.09,
        action_multipliers={
            ACTION_MEDIUM: 1.5,
            ACTION_HARD: 1.4,
            ACTION_EASY: 1.1,
            ACTION_EXPLAIN: 0.6,
            ACTION_EXAMPLE: 0.7,
        },
    ),
}


class StudentSimulator:
    """
    Simulates a student interacting with an adaptive tutoring system.

    Internal state (hidden from the RL agent):
        - true_knowledge[topic]: actual mastery level in [0, 1]
        - engagement: attention / motivation in [0, 1]
        - ability: latent IRT ability parameter θ
    """

    def __init__(
        self,
        n_topics: int = 10,
        profile: StudentProfile = StudentProfile.FAST_LEARNER,
        rng: np.random.Generator = None,
        initial_knowledge_range: Tuple[float, float] = (0.0, 0.3),
    ):
        self.n_topics = n_topics
        self.profile = profile
        self.params = PROFILE_PARAMS[profile]
        self.rng = rng or np.random.default_rng()
        self._init_knowledge_range = initial_knowledge_range
        self.reset()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset(self) -> None:
        lo, hi = self._init_knowledge_range
        self.true_knowledge = self.rng.uniform(lo, hi, size=self.n_topics)
        # Students start without knowledge of advanced topics locked behind prereqs
        for topic, prereqs in TOPIC_PREREQUISITES.items():
            if topic >= self.n_topics:
                continue
            valid_prereqs = [p for p in prereqs if p < self.n_topics]
            if valid_prereqs and all(self.true_knowledge[p] < 0.2 for p in valid_prereqs):
                self.true_knowledge[topic] = 0.0

        self.engagement = self.rng.uniform(0.55, 0.85)
        self.ability = self.rng.normal(0.0, 1.0)   # IRT ability θ
        self.current_topic = 0
        self.difficulty_level = 0.3
        self.consecutive_correct = 0
        self.consecutive_wrong = 0
        self.step_count = 0

    # ------------------------------------------------------------------
    # IRT response model
    # ------------------------------------------------------------------

    def _irt_probability(self, topic: int, difficulty: float) -> float:
        """3-parameter logistic IRT model."""
        discrimination = 1.5
        guessing = 0.1
        ability_effective = self.ability + self.true_knowledge[topic] * 2.0
        exponent = -discrimination * (ability_effective - difficulty * 3 - 1.5)
        return guessing + (1 - guessing) / (1 + np.exp(exponent))

    # ------------------------------------------------------------------
    # Step: process one teaching action
    # ------------------------------------------------------------------

    def step(
        self, action: int, new_topic: int = None
    ) -> Tuple[bool, float, float]:
        """
        Apply a tutoring action and return (answered_correctly, knowledge_gain, engagement_delta).
        answered_correctly is only meaningful for question actions (2, 3, 4).
        """
        self.step_count += 1
        params = self.params
        multiplier = params.action_multipliers.get(action, 1.0)

        answered_correctly = False
        knowledge_gain = 0.0
        engagement_delta = -params.engagement_decay  # natural decay every step

        if action == ACTION_EXPLAIN:
            knowledge_gain = params.learning_rate * 0.8 * multiplier
            engagement_delta += 0.01

        elif action == ACTION_EXAMPLE:
            knowledge_gain = params.learning_rate * 1.0 * multiplier
            engagement_delta += 0.02

        elif action in (ACTION_EASY, ACTION_MEDIUM, ACTION_HARD):
            diff_map = {ACTION_EASY: 0.25, ACTION_MEDIUM: 0.50, ACTION_HARD: 0.80}
            question_diff = diff_map[action]
            p_correct = self._irt_probability(self.current_topic, question_diff)
            answered_correctly = self.rng.random() < p_correct

            if answered_correctly:
                knowledge_gain = params.learning_rate * multiplier * (1 + 0.5 * question_diff)
                engagement_delta += params.engagement_recovery
                self.consecutive_correct += 1
                self.consecutive_wrong = 0
            else:
                knowledge_gain = -params.forgetting_rate * multiplier * 0.5
                engagement_delta -= 0.04
                self.consecutive_wrong += 1
                self.consecutive_correct = 0

        elif action == ACTION_HINT:
            knowledge_gain = params.learning_rate * 0.5 * multiplier
            engagement_delta += 0.01

        elif action == ACTION_INC_DIFF:
            self.difficulty_level = min(1.0, self.difficulty_level + 0.1)
            engagement_delta += 0.005

        elif action == ACTION_DEC_DIFF:
            self.difficulty_level = max(0.0, self.difficulty_level - 0.1)
            engagement_delta += 0.005

        elif action == ACTION_SWITCH:
            if new_topic is not None:
                self.current_topic = new_topic
            engagement_delta += 0.03  # novelty boost

        elif action == ACTION_REVIEW:
            # Review helps consolidate, slight knowledge boost to prerequisites
            for prereq in [p for p in TOPIC_PREREQUISITES.get(self.current_topic, []) if p < self.n_topics]:
                self.true_knowledge[prereq] = min(
                    1.0, self.true_knowledge[prereq] + params.learning_rate * 0.4
                )
            engagement_delta += 0.01

        elif action == ACTION_ENCOURAGE:
            engagement_delta += 0.08

        # Apply knowledge update with ceiling
        self.true_knowledge[self.current_topic] = np.clip(
            self.true_knowledge[self.current_topic] + knowledge_gain, 0.0, 1.0
        )
        # Apply engagement update
        self.engagement = np.clip(self.engagement + engagement_delta, 0.0, 1.0)

        return answered_correctly, knowledge_gain, engagement_delta

    # ------------------------------------------------------------------
    # Observable state (returned to the RL agent)
    # ------------------------------------------------------------------

    def get_observable_state(self, session_progress: float) -> np.ndarray:
        """
        Returns a 26-dimensional observation vector:
          [0:10]  noisy knowledge estimates per topic
          [10:20] one-hot current topic
          [20]    difficulty_level
          [21]    engagement
          [22]    session_progress
          [23]    consecutive_correct / 10 (normalised)
          [24]    consecutive_wrong / 10
          [25]    prerequisite readiness for current topic
        """
        # Agent sees noisy version of true knowledge (simulates lack of perfect observability)
        noise = self.rng.normal(0, 0.05, size=self.n_topics)
        obs_knowledge = np.clip(self.true_knowledge + noise, 0.0, 1.0)

        topic_one_hot = np.zeros(self.n_topics)
        topic_one_hot[self.current_topic] = 1.0

        prereq_readiness = 1.0
        prereqs = TOPIC_PREREQUISITES.get(self.current_topic, [])
        if prereqs:
            prereq_readiness = np.mean([self.true_knowledge[p] for p in prereqs])

        return np.concatenate([
            obs_knowledge,
            topic_one_hot,
            [self.difficulty_level,
             self.engagement,
             session_progress,
             min(self.consecutive_correct / 10.0, 1.0),
             min(self.consecutive_wrong / 10.0, 1.0),
             prereq_readiness],
        ]).astype(np.float32)

    @property
    def state_dim(self) -> int:
        return self.n_topics * 2 + 6  # 26

    def is_disengaged(self) -> bool:
        return self.engagement < 0.08

    def mastered_topics(self, threshold: float = 0.85) -> List[int]:
        return [t for t in range(self.n_topics) if self.true_knowledge[t] >= threshold]
