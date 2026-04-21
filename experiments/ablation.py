"""
Ablation study — compares:
  A) Full system: PPO + Thompson Sampling + Orchestrator
  B) PPO only (no bandit agents — tutorial agent controls everything)
  C) Bandits only (no PPO — heuristic content + assessment bandits)
  D) Random policy baseline

Usage:
    python experiments/ablation.py --timesteps 100000 --eval_eps 20
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.environment.tutorial_env import TutorialEnv
from src.environment.student_simulator import StudentProfile
from src.agents.tutorial_agent import TutorialAgent
from src.agents.assessment_agent import AssessmentAgent
from src.agents.content_agent import ContentAgent
from src.agents.orchestrator import OrchestratorAgent
from src.rl.ppo import PPOConfig
from src.utils.metrics import compute_metrics, EpisodeStats, cohen_d, relative_improvement
from src.utils.visualization import Visualizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=80000)
    p.add_argument("--eval_eps",  type=int, default=20)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--results_dir", default="results")
    return p.parse_args()


def make_full_system(seed: int) -> OrchestratorAgent:
    env = TutorialEnv(n_topics=10, max_steps=50, seed=seed)
    cfg = PPOConfig(lr=3e-4, hidden_dim=256, update_interval=1024)
    tutorial   = TutorialAgent(ppo_config=cfg)
    assessment = AssessmentAgent(alpha=1.0)
    content    = ContentAgent(alpha=1.0)
    return OrchestratorAgent(env, tutorial, assessment, content)


def train(system: OrchestratorAgent, timesteps: int) -> None:
    obs = system.reset()
    for t in range(timesteps):
        obs, _, done, _ = system.run_step(obs)
        if done:
            obs = system.reset()


def evaluate(system: OrchestratorAgent, n_episodes: int) -> list:
    stats = []
    for _ in range(n_episodes):
        system.env._profile = StudentProfile(np.random.randint(0, 4))
        obs = system.reset()
        done = False
        while not done:
            obs, _, done, _ = system.run_step(obs)
        s = system.env.episode_summary()
        stats.append(EpisodeStats(
            total_reward=s["total_reward"],
            mean_knowledge_final=s["mean_knowledge_final"],
            knowledge_gain=s["knowledge_gain"],
            n_mastered=s["n_mastered"],
            steps=s["steps"],
            disengaged=s["disengaged"],
        ))
    return stats


def run_random(env: TutorialEnv, n_episodes: int) -> list:
    stats = []
    for _ in range(n_episodes):
        env.reset()
        done = False
        while not done:
            action = np.random.randint(0, env.action_dim)
            _, _, done, _ = env.step(action)
        s = env.episode_summary()
        stats.append(EpisodeStats(
            total_reward=s["total_reward"],
            mean_knowledge_final=s["mean_knowledge_final"],
            knowledge_gain=s["knowledge_gain"],
            n_mastered=s["n_mastered"],
            steps=s["steps"],
            disengaged=s["disengaged"],
        ))
    return stats


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    viz = Visualizer(save_dir=os.path.join(args.results_dir, "figures"))

    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)

    # A) Full system
    print("\n[A] Training full system (PPO + Thompson Sampling)...")
    sys_full = make_full_system(args.seed)
    train(sys_full, args.timesteps)
    stats_full = evaluate(sys_full, args.eval_eps)
    m_full = compute_metrics(stats_full)

    # B) PPO only (disable bandit routing — orchestrator always uses tutorial agent)
    print("[B] Training PPO-only system...")
    sys_ppo = make_full_system(args.seed + 10)
    # Patch orchestrator to always use PPO
    sys_ppo._mode_override = 0  # force LEARNING mode
    from src.agents.orchestrator import TeachingMode
    original_decide = sys_ppo._decide_mode
    sys_ppo._decide_mode = lambda obs, topic: TeachingMode.LEARNING
    train(sys_ppo, args.timesteps)
    stats_ppo = evaluate(sys_ppo, args.eval_eps)
    m_ppo = compute_metrics(stats_ppo)

    # C) Random baseline
    print("[C] Running random baseline...")
    env_rand = TutorialEnv(n_topics=10, max_steps=50, seed=args.seed + 20)
    stats_rand = run_random(env_rand, args.eval_eps)
    m_rand = compute_metrics(stats_rand)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'System':<30} {'Reward':<16} {'Knowledge':<16} {'Mastered'}")
    print("-" * 70)
    for name, m in [
        ("Full (PPO + TS + Orch)", m_full),
        ("PPO Only",               m_ppo),
        ("Random Baseline",         m_rand),
    ]:
        print(
            f"{name:<30} "
            f"{m['reward_mean']:5.1f}±{m['reward_sem']:.1f}      "
            f"{m['knowledge_mean']:.3f}±{m['knowledge_sem']:.3f}      "
            f"{m['mastered_mean']:.1f}"
        )
    print("=" * 70)

    # Effect sizes
    for name, stats_b in [("PPO Only", stats_ppo), ("Random", stats_rand)]:
        d = cohen_d(
            [s.total_reward for s in stats_full],
            [s.total_reward for s in stats_b],
        )
        ri = relative_improvement(
            [s.total_reward for s in stats_b],
            [s.total_reward for s in stats_full],
        )
        print(f"Full vs {name:<12}: Cohen's d = {d:.2f}, Improvement = {ri:.1f}%")

    # Bar chart
    viz.plot_comparison_bar(
        {
            "Full System": (m_full["reward_mean"],  m_full["reward_sem"]),
            "PPO Only":    (m_ppo["reward_mean"],   m_ppo["reward_sem"]),
            "Random":      (m_rand["reward_mean"],  m_rand["reward_sem"]),
        },
        metric="Mean Episode Reward",
        filename="ablation_reward.png",
    )
    viz.plot_comparison_bar(
        {
            "Full System": (m_full["knowledge_mean"],  m_full["knowledge_sem"]),
            "PPO Only":    (m_ppo["knowledge_mean"],   m_ppo["knowledge_sem"]),
            "Random":      (m_rand["knowledge_mean"],  m_rand["knowledge_sem"]),
        },
        metric="Mean Final Knowledge",
        filename="ablation_knowledge.png",
    )
    print(f"\nAblation plots saved to {viz.save_dir}/")


if __name__ == "__main__":
    main()
