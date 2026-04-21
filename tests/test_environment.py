"""Tests for the tutoring environment and student simulator."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from src.environment.student_simulator import StudentSimulator, StudentProfile
from src.environment.tutorial_env import TutorialEnv, STATE_DIM, ACTION_DIM
from src.environment.reward_function import RewardFunction


class TestStudentSimulator:
    def test_reset_shapes(self):
        sim = StudentSimulator(n_topics=10)
        sim.reset()
        assert sim.true_knowledge.shape == (10,)
        assert 0.0 <= sim.engagement <= 1.0

    def test_step_returns_tuple(self):
        sim = StudentSimulator(n_topics=10, profile=StudentProfile.FAST_LEARNER)
        ans, k_gain, eng_d = sim.step(2)
        assert isinstance(ans, (bool, np.bool_))
        assert isinstance(k_gain, float)
        assert isinstance(eng_d, float)

    def test_knowledge_clipped(self):
        sim = StudentSimulator(n_topics=10)
        for _ in range(200):
            sim.step(np.random.randint(0, 11))
        assert np.all(sim.true_knowledge >= 0.0)
        assert np.all(sim.true_knowledge <= 1.0)

    def test_engagement_clipped(self):
        sim = StudentSimulator(n_topics=10)
        for _ in range(100):
            sim.step(np.random.randint(0, 11))
        assert 0.0 <= sim.engagement <= 1.0

    def test_observable_state_dim(self):
        sim = StudentSimulator(n_topics=10)
        obs = sim.get_observable_state(0.5)
        assert obs.shape == (STATE_DIM,)

    def test_switch_topic(self):
        sim = StudentSimulator(n_topics=10)
        sim.current_topic = 0
        sim.step(8, new_topic=3)
        assert sim.current_topic == 3

    def test_profiles_differ(self):
        """Fast learner should accumulate more knowledge than slow learner."""
        rng = np.random.default_rng(0)
        fast = StudentSimulator(n_topics=10, profile=StudentProfile.FAST_LEARNER, rng=rng)
        rng2 = np.random.default_rng(0)
        slow = StudentSimulator(n_topics=10, profile=StudentProfile.SLOW_LEARNER, rng=rng2)
        # Same actions
        for a in [0, 3, 3, 3, 3] * 10:
            fast.step(a)
            slow.step(a)
        assert np.mean(fast.true_knowledge) > np.mean(slow.true_knowledge)


class TestTutorialEnv:
    def test_reset_returns_obs(self):
        env = TutorialEnv(n_topics=10, max_steps=50)
        obs = env.reset()
        assert obs.shape == (STATE_DIM,)
        assert obs.dtype == np.float32

    def test_step_interface(self):
        env = TutorialEnv()
        env.reset()
        obs, reward, done, info = env.step(3)
        assert obs.shape == (STATE_DIM,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "mean_knowledge" in info

    def test_episode_terminates(self):
        env = TutorialEnv(n_topics=10, max_steps=10)
        env.reset()
        for _ in range(11):
            _, _, done, _ = env.step(3)
            if done:
                break
        assert done

    def test_action_space(self):
        env = TutorialEnv()
        env.reset()
        for a in range(ACTION_DIM):
            env.reset()
            obs, r, done, info = env.step(a)
            assert obs.shape == (STATE_DIM,)

    def test_episode_summary(self):
        env = TutorialEnv(max_steps=5)
        env.reset()
        for _ in range(5):
            _, _, done, _ = env.step(3)
            if done:
                break
        summary = env.episode_summary()
        assert "total_reward" in summary
        assert "mean_knowledge_final" in summary


class TestRewardFunction:
    def test_mastery_bonus(self):
        rf = RewardFunction()
        rf.reset()
        knowledge = np.full(10, 0.0)
        knowledge[0] = 0.9
        r = rf.compute(
            action=3, answered_correctly=True,
            knowledge_gain=0.05, engagement=0.8,
            engagement_delta=0.0, current_topic=0,
            true_knowledge=knowledge, prereq_knowledge=1.0,
        )
        assert r > 0  # mastery bonus triggered

    def test_disengagement_penalty(self):
        rf = RewardFunction()
        rf.reset()
        r = rf.compute(
            action=3, answered_correctly=False,
            knowledge_gain=-0.01, engagement=0.05,
            engagement_delta=-0.1, current_topic=0,
            true_knowledge=np.zeros(10), prereq_knowledge=0.0,
        )
        assert r < 0
