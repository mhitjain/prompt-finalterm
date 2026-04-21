"""
Master experiment script — trains and evaluates all algorithms, generates
all figures, and prints a final comparison table.

Usage:
    python experiments/run_all.py [--timesteps 300000] [--seeds 3]

Output:
    results/
      figures/          ← all PNG plots
      ppo_checkpoints/  ← saved policy weights
      summary.csv       ← per-run metrics
"""

import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.environment.tutorial_env import TutorialEnv
from src.environment.student_simulator import StudentProfile
from src.rl.ppo import PPO, PPOConfig
from src.rl.contextual_bandits import LinThompsonSampling, EpsilonGreedyBandit
from src.agents.tutorial_agent import TutorialAgent
from src.agents.assessment_agent import AssessmentAgent
from src.agents.content_agent import ContentAgent
from src.agents.orchestrator import OrchestratorAgent
from src.utils.metrics import compute_metrics, EpisodeStats, cohen_d, relative_improvement
from src.utils.visualization import Visualizer
from src.tools.performance_tracker import PerformanceTrackerTool


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=150000)
    p.add_argument("--seeds",     type=int, default=2)
    p.add_argument("--eval_eps",  type=int, default=30)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(seed: int = 42, profile: StudentProfile = None) -> TutorialEnv:
    return TutorialEnv(n_topics=10, max_steps=50, profile=profile, seed=seed)


def make_system(env: TutorialEnv, seed: int = 42) -> OrchestratorAgent:
    np.random.seed(seed)
    cfg = PPOConfig(
        lr=3e-4, gamma=0.99, lambda_gae=0.95,
        clip_epsilon=0.2, epochs=4, batch_size=64,
        value_coef=0.5, entropy_coef=0.01,
        hidden_dim=256, update_interval=1024,
    )
    tutorial    = TutorialAgent(ppo_config=cfg)
    assessment  = AssessmentAgent(alpha=1.0)
    content     = ContentAgent(alpha=1.0)
    orchestrator = OrchestratorAgent(
        env=env, tutorial_agent=tutorial,
        assessment_agent=assessment, content_agent=content,
        verbose=False,
    )
    return orchestrator


def evaluate_system(
    orchestrator: OrchestratorAgent,
    n_episodes: int = 30,
    randomise_profiles: bool = True,
) -> list[EpisodeStats]:
    stats = []
    for _ in range(n_episodes):
        profile = (
            StudentProfile(np.random.randint(0, 4))
            if randomise_profiles else None
        )
        orchestrator.env.reset(profile)
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
        ))
    return stats


# ---------------------------------------------------------------------------
# Baseline: random policy
# ---------------------------------------------------------------------------

def run_random_baseline(env: TutorialEnv, n_episodes: int = 30) -> list[EpisodeStats]:
    stats = []
    for _ in range(n_episodes):
        env.reset()
        done = False
        while not done:
            action = np.random.randint(0, env.action_dim)
            _, _, done, _ = env.step(action)
        summary = env.episode_summary()
        stats.append(EpisodeStats(
            total_reward=summary["total_reward"],
            mean_knowledge_final=summary["mean_knowledge_final"],
            knowledge_gain=summary["knowledge_gain"],
            n_mastered=summary["n_mastered"],
            steps=summary["steps"],
            disengaged=summary["disengaged"],
        ))
    return stats


# ---------------------------------------------------------------------------
# Baseline: greedy heuristic (always ask medium question)
# ---------------------------------------------------------------------------

def run_heuristic_baseline(env: TutorialEnv, n_episodes: int = 30) -> list[EpisodeStats]:
    """
    Heuristic: always ASK_MEDIUM (action 3) — represents a static policy
    that doesn't adapt to the student.
    """
    stats = []
    for _ in range(n_episodes):
        env.reset()
        done = False
        step = 0
        while not done:
            # Alternate between explain (0), medium question (3), hint (5)
            action = [0, 3, 5][step % 3]
            _, _, done, _ = env.step(action)
            step += 1
        summary = env.episode_summary()
        stats.append(EpisodeStats(
            total_reward=summary["total_reward"],
            mean_knowledge_final=summary["mean_knowledge_final"],
            knowledge_gain=summary["knowledge_gain"],
            n_mastered=summary["n_mastered"],
            steps=summary["steps"],
            disengaged=summary["disengaged"],
        ))
    return stats


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_and_eval(
    timesteps: int,
    seed: int,
    eval_interval: int,
    n_eval_eps: int,
    results_dir: str,
    verbose: bool = False,
) -> dict:
    """Train the full system for `timesteps` steps, periodically evaluate."""
    env = make_env(seed=seed)
    system = make_system(env, seed=seed)
    tracker = PerformanceTrackerTool()

    episode_curves = {
        "total_reward": [], "mean_knowledge_final": [],
        "knowledge_gain": [], "n_mastered": [],
    }

    t = 0
    episode = 0
    obs = system.reset()

    print(f"\n[Seed {seed}] Training for {timesteps:,} timesteps...")
    t_start = time.time()

    while t < timesteps:
        # Run one environment step via orchestrator
        next_obs, reward, done, info = system.run_step(obs)
        t += 1
        obs = next_obs

        if done:
            summary = env.episode_summary()
            tracker.record_episode(episode, summary)
            for k in episode_curves:
                episode_curves[k].append(summary.get(k, 0.0))
            episode += 1
            obs = system.reset()

            if verbose and episode % 50 == 0:
                print(
                    f"  ep={episode:4d}  t={t:7d}  "
                    f"reward={summary['total_reward']:6.1f}  "
                    f"knowledge={summary['mean_knowledge_final']:.3f}"
                )

    elapsed = time.time() - t_start
    print(f"[Seed {seed}] Training done in {elapsed:.0f}s ({t:,} steps, {episode} episodes)")

    # Final evaluation
    print(f"[Seed {seed}] Evaluating on {n_eval_eps} episodes...")
    eval_stats = evaluate_system(system, n_episodes=n_eval_eps)
    metrics = compute_metrics(eval_stats)

    # Save curves for plotting
    curves_smooth = tracker.get_learning_curves(window=30)

    # Save PPO checkpoint
    ckpt_dir = os.path.join(results_dir, "ppo_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    system.tutorial_agent.save(os.path.join(ckpt_dir, f"ppo_seed{seed}.pt"))

    return {
        "seed": seed,
        "metrics": metrics,
        "curves": curves_smooth,
        "ppo_diagnostics": system.tutorial_agent.ppo.metrics_history,
        "mode_dist": tracker.mode_distribution(),
        "assessment_stats": system.assessment_agent.get_stats(),
        "content_stats": system.content_agent.get_stats(),
        "episode_df": tracker.episode_dataframe(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    viz = Visualizer(save_dir=os.path.join(args.results_dir, "figures"))

    all_runs = []
    for seed in range(args.seeds):
        run = train_and_eval(
            timesteps=args.timesteps,
            seed=seed,
            eval_interval=5000,
            n_eval_eps=args.eval_eps,
            results_dir=args.results_dir,
            verbose=args.verbose,
        )
        all_runs.append(run)

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    print("\nRunning baselines...")
    env_baseline = make_env(seed=99)
    random_stats    = run_random_baseline(env_baseline, n_episodes=args.eval_eps)
    heuristic_stats = run_heuristic_baseline(env_baseline, n_episodes=args.eval_eps)
    random_metrics    = compute_metrics(random_stats)
    heuristic_metrics = compute_metrics(heuristic_stats)

    # Combine PPO results across seeds
    ppo_rewards = []
    ppo_knowledge = []
    for run in all_runs:
        m = run["metrics"]
        ppo_rewards.extend([m["reward_mean"]] * m["n_episodes"])
        ppo_knowledge.extend([m["knowledge_mean"]] * m["n_episodes"])

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    header = f"{'Algorithm':<28} {'Reward (mean±sem)':<22} {'Knowledge (mean±sem)':<22} {'Mastered':<10}"
    print("\n" + "=" * 84)
    print("RESULTS COMPARISON")
    print("=" * 84)
    print(header)
    print("-" * 84)

    def _fmt(m, r_key="reward_mean", r_sem="reward_sem", k_key="knowledge_mean", k_sem="knowledge_sem"):
        return (
            f"{m[r_key]:6.1f} ± {m[r_sem]:.2f}       "
            f"{m[k_key]:.3f} ± {m[k_sem]:.3f}       "
            f"{m['mastered_mean']:.1f}"
        )

    # Aggregate PPO across seeds
    ppo_all_metrics = [r["metrics"] for r in all_runs]
    ppo_agg_reward = np.mean([m["reward_mean"] for m in ppo_all_metrics])
    ppo_agg_r_sem  = np.std([m["reward_mean"] for m in ppo_all_metrics]) / max(1, np.sqrt(len(ppo_all_metrics)))
    ppo_agg_know   = np.mean([m["knowledge_mean"] for m in ppo_all_metrics])
    ppo_agg_k_sem  = np.std([m["knowledge_mean"] for m in ppo_all_metrics]) / max(1, np.sqrt(len(ppo_all_metrics)))
    ppo_agg_mast   = np.mean([m["mastered_mean"] for m in ppo_all_metrics])

    print(f"{'PPO + Thompson Sampling':<28} {ppo_agg_reward:6.1f} ± {ppo_agg_r_sem:.2f}       {ppo_agg_know:.3f} ± {ppo_agg_k_sem:.3f}       {ppo_agg_mast:.1f}")
    print(f"{'Heuristic (Explain+Ask+Hint)':<28} {_fmt(heuristic_metrics)}")
    print(f"{'Random Policy':<28} {_fmt(random_metrics)}")
    print("=" * 84)

    rand_rewards = [s.total_reward for s in random_stats]
    ppo_rewards_flat = []
    for run in all_runs:
        if "episode_df" in run and len(run["episode_df"]) > 0:
            df = run["episode_df"]
            if "total_reward" in df.columns:
                ppo_rewards_flat.extend(df["total_reward"].tail(30).tolist())

    if ppo_rewards_flat and rand_rewards:
        d = cohen_d(ppo_rewards_flat, rand_rewards)
        ri = relative_improvement(rand_rewards, ppo_rewards_flat)
        print(f"\nPPO vs Random: Cohen's d = {d:.2f}, Relative improvement = {ri:.1f}%")

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    print("\nGenerating figures...")

    # Learning curves (averaged over seeds)
    if all_runs and all_runs[0]["curves"]:
        curves_to_plot = {}
        for metric in ["mean_knowledge_final", "total_reward"]:
            for label, run in zip([f"PPO seed {r['seed']}" for r in all_runs], all_runs):
                if run["curves"] and metric in run["curves"]:
                    curves_to_plot[label] = run["curves"]
                    break

        if curves_to_plot:
            viz.plot_learning_curves(
                curves_to_plot, metric="mean_knowledge_final",
                title="PPO Learning Curve — Mean Student Knowledge",
                filename="learning_curve_knowledge.png",
            )
            viz.plot_learning_curves(
                curves_to_plot, metric="total_reward",
                title="PPO Learning Curve — Episode Reward",
                filename="learning_curve_reward.png",
            )

    # Comparison bar chart
    viz.plot_comparison_bar(
        {
            "PPO + TS": (ppo_agg_reward, ppo_agg_r_sem),
            "Heuristic": (heuristic_metrics["reward_mean"], heuristic_metrics["reward_sem"]),
            "Random":    (random_metrics["reward_mean"],    random_metrics["reward_sem"]),
        },
        metric="Mean Episode Reward",
        filename="comparison_reward.png",
    )
    viz.plot_comparison_bar(
        {
            "PPO + TS": (ppo_agg_know, ppo_agg_k_sem),
            "Heuristic": (heuristic_metrics["knowledge_mean"], heuristic_metrics["knowledge_sem"]),
            "Random":    (random_metrics["knowledge_mean"],    random_metrics["knowledge_sem"]),
        },
        metric="Mean Final Knowledge",
        filename="comparison_knowledge.png",
    )

    # PPO diagnostics (first seed)
    if all_runs[0]["ppo_diagnostics"]:
        viz.plot_ppo_diagnostics(
            all_runs[0]["ppo_diagnostics"],
            filename="ppo_diagnostics.png",
        )

    # Mode distribution
    if all_runs[0]["mode_dist"]:
        viz.plot_mode_distribution(
            all_runs[0]["mode_dist"],
            filename="mode_distribution.png",
        )

    # Assessment bandit arm distribution
    a_stats = all_runs[0]["assessment_stats"].get("arm_stats", {})
    if a_stats:
        arm_counts = {arm: int(info["n_pulls"]) for arm, info in a_stats.items()}
        viz.plot_bandit_arm_distribution(
            arm_counts, title="Assessment Bandit Arm Selections",
            filename="assessment_bandit_arms.png",
        )

    print(f"\nAll figures saved to {viz.save_dir}/")

    # ------------------------------------------------------------------
    # Save summary JSON
    # ------------------------------------------------------------------
    summary = {
        "ppo": {"reward_mean": ppo_agg_reward, "knowledge_mean": ppo_agg_know, "mastered_mean": ppo_agg_mast},
        "heuristic": {"reward_mean": heuristic_metrics["reward_mean"], "knowledge_mean": heuristic_metrics["knowledge_mean"]},
        "random": {"reward_mean": random_metrics["reward_mean"], "knowledge_mean": random_metrics["knowledge_mean"]},
    }
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {args.results_dir}/summary.json")


if __name__ == "__main__":
    main()
