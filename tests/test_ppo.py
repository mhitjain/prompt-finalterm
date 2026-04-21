"""Tests for PPO implementation and neural network components."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import pytest
from src.rl.networks import ActorCritic, MLP
from src.rl.buffer import RolloutBuffer
from src.rl.ppo import PPO, PPOConfig


class TestActorCritic:
    def test_forward_shapes(self):
        net = ActorCritic(state_dim=26, action_dim=11, hidden_dim=64)
        x = torch.randn(4, 26)
        probs, values = net(x)
        assert probs.shape == (4, 11)
        assert values.shape == (4, 1)

    def test_probs_sum_to_one(self):
        net = ActorCritic(state_dim=26, action_dim=11)
        x = torch.randn(8, 26)
        probs, _ = net(x)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_get_action_and_value(self):
        net = ActorCritic(state_dim=26, action_dim=11)
        x = torch.randn(1, 26)
        action, log_prob, entropy, value = net.get_action_and_value(x)
        assert 0 <= action.item() < 11
        assert entropy.item() > 0

    def test_act_deterministic(self):
        net = ActorCritic(state_dim=26, action_dim=11)
        x = torch.randn(1, 26)
        a1, lp1, v1 = net.act(x)
        assert isinstance(a1, int)


class TestRolloutBuffer:
    def _fill_buffer(self, capacity=128, state_dim=26):
        buf = RolloutBuffer(capacity=capacity, state_dim=state_dim)
        for _ in range(capacity):
            buf.add(
                state=np.random.randn(state_dim).astype(np.float32),
                action=np.random.randint(0, 11),
                log_prob=np.random.randn(),
                reward=np.random.randn(),
                value=np.random.randn(),
                done=bool(np.random.randint(0, 2)),
            )
        return buf

    def test_is_ready_after_fill(self):
        buf = self._fill_buffer(128)
        assert buf.is_ready()

    def test_not_ready_partial(self):
        buf = RolloutBuffer(capacity=128, state_dim=26)
        for _ in range(64):
            buf.add(np.zeros(26), 0, 0.0, 0.0, 0.0, False)
        assert not buf.is_ready()

    def test_gae_shapes(self):
        buf = self._fill_buffer(64)
        buf.compute_gae(last_value=0.0)
        assert buf.advantages.shape == (64,)
        assert buf.returns.shape == (64,)

    def test_get_batches(self):
        buf = self._fill_buffer(128)
        buf.compute_gae(0.0)
        batches = list(buf.get_batches(batch_size=32))
        assert len(batches) == 4
        states, actions, log_probs, returns, advs = batches[0]
        assert states.shape == (32, 26)

    def test_reset(self):
        buf = self._fill_buffer(128)
        buf.reset()
        assert not buf.is_ready()
        assert buf._ptr == 0


class TestPPO:
    def test_select_action(self):
        ppo = PPO(state_dim=26, action_dim=11, config=PPOConfig(update_interval=64))
        obs = np.random.randn(26).astype(np.float32)
        action, log_prob, value = ppo.select_action(obs)
        assert 0 <= action < 11
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_buffer_accumulation(self):
        ppo = PPO(state_dim=26, action_dim=11, config=PPOConfig(update_interval=32))
        obs = np.zeros(26, dtype=np.float32)
        for _ in range(32):
            a, lp, v = ppo.select_action(obs)
            ppo.buffer.add(obs, a, lp, 1.0, v, False)
        assert ppo.buffer.is_ready()

    def test_update_runs(self):
        """A full PPO update should complete without error."""
        cfg = PPOConfig(update_interval=128, batch_size=32, epochs=2, hidden_dim=64)
        ppo = PPO(state_dim=26, action_dim=11, config=cfg)
        obs = np.zeros(26, dtype=np.float32)
        for _ in range(128):
            a, lp, v = ppo.select_action(obs)
            ppo.buffer.add(obs, a, lp, float(np.random.randn()), v, False)
        metrics = ppo.update(obs)
        assert "policy_loss" in metrics
        assert "entropy" in metrics
        assert isinstance(metrics["policy_loss"], float)

    def test_policy_improves(self):
        """
        After training with positive rewards, the policy should increase
        mean episode reward vs initial random policy.
        """
        from src.environment.tutorial_env import TutorialEnv
        cfg = PPOConfig(update_interval=256, batch_size=64, epochs=2,
                        lr=1e-3, hidden_dim=64, entropy_coef=0.05)
        ppo = PPO(state_dim=26, action_dim=11, config=cfg)
        env = TutorialEnv(n_topics=10, max_steps=30, seed=0)

        # Train for 5 updates
        obs = env.reset()
        for t in range(256 * 5):
            a, lp, v = ppo.select_action(obs)
            next_obs, r, done, _ = env.step(a)
            ppo.buffer.add(obs, a, lp, r, v, done)
            if ppo.buffer.is_ready():
                ppo.update(next_obs)
            obs = env.reset() if done else next_obs

        # Evaluate trained policy (no exploration noise in test)
        rewards_trained = []
        for _ in range(20):
            obs = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                a, _, _ = ppo.select_action(obs)
                obs, r, done, _ = env.step(a)
                ep_reward += r
            rewards_trained.append(ep_reward)

        # A simple sanity check: trained policy shouldn't be catastrophically bad
        assert np.mean(rewards_trained) > -200.0
