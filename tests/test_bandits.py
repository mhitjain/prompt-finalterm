"""Tests for Thompson Sampling contextual bandit."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from src.rl.contextual_bandits import LinThompsonSampling, EpsilonGreedyBandit


class TestLinThompsonSampling:
    def test_select_returns_valid_arm(self):
        bandit = LinThompsonSampling(n_arms=3, context_dim=5)
        ctx = np.random.randn(5)
        arm = bandit.select_arm(ctx)
        assert 0 <= arm < 3

    def test_update_changes_stats(self):
        bandit = LinThompsonSampling(n_arms=3, context_dim=5)
        ctx = np.ones(5)
        bandit.update(0, ctx, reward=1.0)
        assert bandit.stats[0].n_pulls == 1
        assert bandit.stats[0].total_reward == pytest.approx(1.0)

    def test_posterior_mean_approaches_true(self):
        """
        With a true reward function r(ctx, arm) = ctx @ theta_true,
        the posterior mean should converge toward theta_true for the correct arm.
        """
        np.random.seed(42)
        d = 5
        theta_true = np.array([1.0, 0.5, -0.5, 0.2, 0.1])
        bandit = LinThompsonSampling(n_arms=1, context_dim=d, alpha=0.1)

        for _ in range(500):
            ctx = np.random.randn(d)
            reward = float(ctx @ theta_true + np.random.randn() * 0.1)
            bandit.update(0, ctx, reward)

        # Posterior mean should be close to theta_true
        diff = np.linalg.norm(bandit.mu[0] - theta_true)
        assert diff < 1.5, f"Posterior mean not converging: diff={diff:.3f}"

    def test_good_arm_selected_more_often(self):
        """
        If arm 0 always gives high reward and arm 1 always gives low reward,
        Thompson Sampling should select arm 0 much more after learning.
        """
        np.random.seed(0)
        d = 4
        bandit = LinThompsonSampling(n_arms=2, context_dim=d, alpha=1.0)
        ctx = np.ones(d)

        # Burn-in: arm 0 always +2, arm 1 always -1
        for _ in range(100):
            bandit.update(0, ctx, 2.0)
            bandit.update(1, ctx, -1.0)

        # Count arm selections over 200 steps
        counts = [0, 0]
        for _ in range(200):
            arm = bandit.select_arm(ctx)
            counts[arm] += 1

        # Arm 0 should be preferred
        assert counts[0] > counts[1], f"Expected arm 0 preferred, got counts={counts}"

    def test_ucb_warmup(self):
        bandit = LinThompsonSampling(n_arms=3, context_dim=6)
        ctx = np.random.randn(6)
        arm = bandit.select_arm_ucb(ctx)
        assert 0 <= arm < 3

    def test_reset(self):
        bandit = LinThompsonSampling(n_arms=3, context_dim=5)
        for _ in range(20):
            bandit.update(0, np.ones(5), 1.0)
        bandit.reset()
        assert bandit.stats[0].n_pulls == 0

    def test_arm_summary(self):
        bandit = LinThompsonSampling(n_arms=2, context_dim=4)
        bandit.update(0, np.ones(4), 0.5)
        summary = bandit.arm_summary()
        assert 0 in summary and 1 in summary
        assert summary[0]["n_pulls"] == 1


class TestEpsilonGreedyBandit:
    def test_select_valid_arm(self):
        b = EpsilonGreedyBandit(n_arms=5, epsilon=0.1)
        arm = b.select_arm()
        assert 0 <= arm < 5

    def test_learns_best_arm(self):
        np.random.seed(1)
        b = EpsilonGreedyBandit(n_arms=3, epsilon=0.05)
        for _ in range(300):
            b.update(0, None, 1.0)
            b.update(1, None, 0.1)
            b.update(2, None, 0.5)
        assert b.select_arm() == 0
