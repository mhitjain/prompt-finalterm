"""
Evaluation script — loads a trained PPO checkpoint and evaluates it across
all student profiles, producing per-profile and aggregate statistics.

Usage:
    python experiments/evaluate.py --checkpoint results/ppo_checkpoints/ppo_seed0.pt
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.environment.tutorial_env import TutorialEnv
from src.environment.student_simulator import StudentProfile, TOPIC_NAMES
from src.agents.tutorial_agent import TutorialAgent
from src.agents.assessment_agent import AssessmentAgent
from src.agents.content_agent import ContentAgent
from src.agents.orchestrator import OrchestratorAgent
from src.rl.ppo import PPOConfig
from src.utils.metrics import compute_metrics, EpisodeStats
from src.utils.visualization import Visualizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--n_episodes", type=int, default=20)
    p.add_argument("--results_dir", type=str, default="results")
    return p.parse_args()


def eval_per_profile(
    orchestrator: OrchestratorAgent,
    n_episodes: int,
    checkpoint: str = None,
) -> dict:
    if checkpoint and os.path.exists(checkpoint):
        orchestrator.tutorial_agent.load(checkpoint)
        print(f"Loaded checkpoint: {checkpoint}")

    results = {}
    profile_knowledge = {}

    for profile in StudentProfile:
        stats = []
        knowledge_finals = []
        for _ in range(n_episodes):
            orchestrator.env._profile = profile
            obs = orchestrator.reset()
            done = False
            while not done:
                obs, _, done, _ = orchestrator.run_step(obs)
            summary = orchestrator.env.episode_summary()
            stats.append(EpisodeStats(
                total_reward=summary["total_reward"],
                mean_knowledge_final=summary["mean_knowledge_final"],
                knowledge_gain=summary["knowledge_gain"],
                n_mastered=summary["n_mastered"],
                steps=summary["steps"],
                disengaged=summary["disengaged"],
                profile=int(profile),
            ))
            knowledge_finals.append(orchestrator.env.student.true_knowledge.copy())

        metrics = compute_metrics(stats)
        results[profile.name] = metrics
        profile_knowledge[profile.name] = np.mean(knowledge_finals, axis=0)
        print(
            f"  {profile.name:<18}  "
            f"reward={metrics['reward_mean']:6.1f}±{metrics['reward_sem']:.1f}  "
            f"knowledge={metrics['knowledge_mean']:.3f}±{metrics['knowledge_sem']:.3f}  "
            f"mastered={metrics['mastered_mean']:.1f}"
        )

    return results, profile_knowledge


def main():
    args = parse_args()
    viz = Visualizer(save_dir=os.path.join(args.results_dir, "figures"))

    env = TutorialEnv(n_topics=10, max_steps=50, seed=77)
    tutorial    = TutorialAgent(ppo_config=PPOConfig(hidden_dim=256, update_interval=1024))
    assessment  = AssessmentAgent(alpha=1.0)
    content     = ContentAgent(alpha=1.0)
    orchestrator = OrchestratorAgent(env, tutorial, assessment, content)

    print("\nEvaluating per student profile:")
    print("-" * 70)
    results, profile_knowledge = eval_per_profile(
        orchestrator, n_episodes=args.n_episodes, checkpoint=args.checkpoint
    )

    # Knowledge heatmap
    path = viz.plot_knowledge_heatmap(
        profile_knowledge=profile_knowledge,
        topic_names=TOPIC_NAMES,
        filename="knowledge_heatmap.png",
    )
    print(f"\nKnowledge heatmap saved to {path}")

    # Overall aggregate
    all_stats = []
    for profile_name, metrics in results.items():
        # Reconstruct as stats list from metrics (approximate)
        all_stats.append(EpisodeStats(
            total_reward=metrics["reward_mean"],
            mean_knowledge_final=metrics["knowledge_mean"],
            knowledge_gain=metrics["k_gain_mean"],
            n_mastered=int(metrics["mastered_mean"]),
            steps=50,
            disengaged=False,
        ))
    agg = compute_metrics(all_stats)
    print(f"\nAggregate across profiles:")
    print(f"  Reward:    {agg['reward_mean']:.2f} ± {agg['reward_sem']:.2f}")
    print(f"  Knowledge: {agg['knowledge_mean']:.3f} ± {agg['knowledge_sem']:.3f}")
    print(f"  Mastered:  {agg['mastered_mean']:.1f} ± {agg['mastered_sem']:.1f}")


if __name__ == "__main__":
    main()
