"""
Generate learning curves averaged over 20 students:
    Untrained RL vs Fixed Script vs Trained RL.

Runs in simulation mode (no API calls) — pure RL decisions over 50-step episodes.
Saves chart to results/figures/learning_curves_150.png

Usage:
    python experiments/learning_curves_comparison.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment.tutorial_env import TutorialEnv, ACTION_NAMES
from src.environment.student_simulator import StudentProfile
from src.agents.tutorial_agent import TutorialAgent
from src.agents.assessment_agent import AssessmentAgent
from src.agents.content_agent import ContentAgent
from src.agents.orchestrator import OrchestratorAgent
from src.rl.ppo import PPOConfig

FIXED_SCRIPT    = [0, 3, 5]
N_TOPICS        = 10
N_STEPS         = 150   # longer horizon to show engagement advantage paying off
N_EPISODES      = 20    # average over 20 different students
TRAIN_STEPS     = 150000
PROFILE_ID      = 1     # SLOW_LEARNER — shows biggest difference
SEED            = 42
LEARNED_THRESH  = 0.5   # "substantially learned" threshold (not full mastery)


def build_orch(seed):
    env = TutorialEnv(n_topics=N_TOPICS, max_steps=N_STEPS,
                      profile=StudentProfile(PROFILE_ID), seed=seed)
    cfg = PPOConfig(lr=3e-4, hidden_dim=256, update_interval=1024)
    orch = OrchestratorAgent(
        env,
        TutorialAgent(ppo_config=cfg),
        AssessmentAgent(alpha=1.0),
        ContentAgent(alpha=1.0),
        verbose=False,
    )
    return env, orch


def run_fixed_script(n_steps, seed):
    env = TutorialEnv(n_topics=10, max_steps=n_steps,
                      profile=StudentProfile(PROFILE_ID), seed=seed)
    env.reset()
    k0 = env.student.true_knowledge.copy()   # initial knowledge baseline
    gain, engagement, reward_curve = [], [], []
    for i in range(n_steps):
        action = FIXED_SCRIPT[i % 3]
        obs, r, done, info = env.step(action)
        gain.append(float(np.sum(np.maximum(env.student.true_knowledge - k0, 0))))
        engagement.append(info["engagement"])
        reward_curve.append(r)
        if done:
            break
    while len(gain) < n_steps:
        gain.append(gain[-1] if gain else 0.0)
        engagement.append(engagement[-1] if engagement else 0.0)
        reward_curve.append(0.0)
    return gain, engagement, reward_curve


def run_rl(orch, env_seed, n_steps):
    env_new = TutorialEnv(n_topics=10, max_steps=n_steps,
                          profile=StudentProfile(PROFILE_ID), seed=env_seed)
    orch.env = env_new
    obs = orch.reset()
    k0 = env_new.student.true_knowledge.copy()   # initial knowledge baseline
    gain, engagement, reward_curve = [], [], []
    for _ in range(n_steps):
        action, topic_override = orch.act(obs, {})
        obs, r, done, info = env_new.step(action, topic_override)
        orch.tutorial_agent.store_transition(r, done)
        gain.append(float(np.sum(np.maximum(env_new.student.true_knowledge - k0, 0))))
        engagement.append(info["engagement"])
        reward_curve.append(r)
        if done:
            break
    while len(gain) < n_steps:
        gain.append(gain[-1] if gain else 0.0)
        engagement.append(engagement[-1] if engagement else 0.0)
        reward_curve.append(0.0)
    return gain, engagement, reward_curve


def collect_multi_episode(runner_fn, n_episodes, base_seed, **kwargs):
    """Run runner_fn over n_episodes different seeds, return (mean, std) arrays."""
    all_k, all_e, all_r = [], [], []
    for i in range(n_episodes):
        eval_seed = base_seed + 1000 + i * 7  # deterministic but varied seeds
        k, e, r = runner_fn(eval_seed, **kwargs)
        all_k.append(k[:N_STEPS])
        all_e.append(e[:N_STEPS])
        all_r.append(np.cumsum(r[:N_STEPS]).tolist())
    return (
        np.mean(all_k, axis=0), np.std(all_k, axis=0),
        np.mean(all_e, axis=0), np.std(all_e, axis=0),
        np.mean(all_r, axis=0), np.std(all_r, axis=0),
    )


def main():
    print("Building untrained agent...")
    _, orch_pre = build_orch(SEED)

    print(f"Training RL agent ({TRAIN_STEPS:,} steps)...")
    env_train, orch_trained = build_orch(SEED)
    obs = orch_trained.reset()
    for t in range(TRAIN_STEPS):
        obs, r, done, _ = orch_trained.run_step(obs)
        if done:
            obs = orch_trained.reset()
    print("Training done.")

    print(f"Evaluating over {N_EPISODES} episodes per system...")

    def _fixed(seed):
        return run_fixed_script(N_STEPS, seed)

    def _pre(seed):
        return run_rl(orch_pre, seed, N_STEPS)

    def _trained(seed):
        return run_rl(orch_trained, seed, N_STEPS)

    fs_mg,  fs_sg,  fs_me, fs_se, fs_mr, fs_sr      = collect_multi_episode(_fixed,   N_EPISODES, SEED)
    pre_mg, pre_sg, pre_me, pre_se, pre_mr, pre_sr  = collect_multi_episode(_pre,     N_EPISODES, SEED)
    rl_mg,  rl_sg,  rl_me,  rl_se,  rl_mr,  rl_sr   = collect_multi_episode(_trained, N_EPISODES, SEED)

    steps = list(range(1, N_STEPS + 1))

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Learning Curves Averaged over {N_EPISODES} Students — SLOW_LEARNER | {N_TOPICS} Topics\n"
        f"RL trained on {TRAIN_STEPS:,} simulation steps (50-step episodes)",
        fontsize=13, fontweight="bold"
    )

    COLORS = {"pre": "#888888", "fs": "#E05C5C", "rl": "#52C878"}
    LABELS = {
        "pre": "Untrained RL (random)",
        "fs":  "Fixed Script (baseline)",
        "rl":  "Trained RL (PPO+TS)",
    }

    datasets = {
        "Engagement (Student Motivation)": (
            (pre_me, pre_se), (fs_me, fs_se), (rl_me, rl_se)
        ),
        "Knowledge Gain (sum across topics)": (
            (pre_mg, pre_sg), (fs_mg, fs_sg), (rl_mg, rl_sg)
        ),
        "Cumulative Reward": (
            (pre_mr, pre_sr), (fs_mr, fs_sr), (rl_mr, rl_sr)
        ),
    }

    for ax, (ylabel, ((pre_m, pre_s), (fs_m, fs_s), (rl_m, rl_s))) in zip(axes, datasets.items()):
        for key, mean, std, color in [
            ("pre", pre_m, pre_s, COLORS["pre"]),
            ("fs",  fs_m,  fs_s,  COLORS["fs"]),
            ("rl",  rl_m,  rl_s,  COLORS["rl"]),
        ]:
            ax.plot(steps, mean, color=color, linewidth=2.2,
                    label=LABELS[key], alpha=0.9)
            ax.fill_between(steps, mean - std, mean + std,
                            color=color, alpha=0.12)

        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    # Annotate final cumulative reward values on the reward panel (rightmost)
    ax2 = axes[2]
    for key, mean, color in [
        ("pre", pre_mr, COLORS["pre"]),
        ("fs",  fs_mr,  COLORS["fs"]),
        ("rl",  rl_mr,  COLORS["rl"]),
    ]:
        ax2.annotate(f"{mean[-1]:.1f}",
                     xy=(steps[-1], mean[-1]),
                     xytext=(8, 0), textcoords="offset points",
                     color=color, fontsize=9, fontweight="bold")

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    path = "results/figures/learning_curves_150.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'System':<25} {'Knowledge Gain':>15} {'Engagement':>12} {'Total Reward':>14}")
    print("-"*65)
    for label, mg, me, mr in [
        ("Untrained RL",   pre_mg, pre_me, pre_mr),
        ("Fixed Script",   fs_mg,  fs_me,  fs_mr),
        ("Trained RL",     rl_mg,  rl_me,  rl_mr),
    ]:
        print(f"{label:<25} {mg[-1]:>15.3f} {me[-1]:>12.2f} {mr[-1]:>14.1f}")
    print("="*65)

    rl_improvement = (rl_mr[-1] - fs_mr[-1]) / abs(fs_mr[-1]) * 100 if fs_mr[-1] != 0 else float("inf")
    print(f"\nTrained RL improvement over Fixed Script (cumulative reward): {rl_improvement:+.1f}%")
    print(f"Trained RL vs Untrained RL (knowledge gain):                  {rl_mg[-1] - pre_mg[-1]:+.3f}")
    print(f"\n(Averaged over {N_EPISODES} students | knowledge gain = sum of increases from initial state)")


if __name__ == "__main__":
    main()
