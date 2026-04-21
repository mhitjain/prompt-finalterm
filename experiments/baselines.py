"""
Educational Baselines — rigorous comparison against established ITS algorithms.

Baselines implemented:
  1. Random Policy          — uniform random action selection
  2. Fixed Script           — rigid explain → ask → hint cycle
  3. Mastery Learning       — advance only when student hits 80% accuracy
  4. Zone of Proximal Dev.  — always target 70% success rate (optimal challenge)
  5. Spaced Repetition      — Leitner box system for review scheduling
  6. Our PPO + TS system    — trained RL agent

These are all well-known ITS algorithms from the educational technology literature.
Comparing against them is what separates rigorous analysis from toy benchmarks.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from typing import List
from src.environment.tutorial_env import TutorialEnv
from src.environment.student_simulator import StudentProfile
from src.utils.metrics import EpisodeStats, compute_metrics, cohen_d, relative_improvement


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_policy(policy_fn, n_episodes: int = 40, seed: int = 0) -> List[EpisodeStats]:
    stats = []
    rng = np.random.default_rng(seed)
    for ep in range(n_episodes):
        profile = StudentProfile(int(rng.integers(0, 4)))
        env = TutorialEnv(n_topics=10, max_steps=50, profile=profile, seed=seed + ep)
        env.reset()
        done = False
        policy_state = {}
        while not done:
            action = policy_fn(env, policy_state)
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


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: Random
# ─────────────────────────────────────────────────────────────────────────────

def random_policy(env, state):
    return np.random.randint(0, env.action_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: Fixed Script (Explain → Ask Medium → Hint, repeat)
# ─────────────────────────────────────────────────────────────────────────────

def fixed_script_policy(env, state):
    step = state.get("step", 0)
    state["step"] = step + 1
    return [0, 3, 5][step % 3]   # EXPLAIN, ASK_MEDIUM, HINT


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3: Mastery Learning
# Concept: only advance to harder material once student reaches 80% accuracy.
# Implementation: track rolling accuracy; if < 0.8 → ask easy; if ≥ 0.8 → harder.
# Reference: Bloom (1984) "The 2 Sigma Problem"
# ─────────────────────────────────────────────────────────────────────────────

def mastery_learning_policy(env, state):
    if "correct_hist" not in state:
        state["correct_hist"] = []
    obs = env._get_obs()
    consec_correct = obs[23] * 10
    rolling_acc = np.mean(state["correct_hist"][-5:]) if state["correct_hist"] else 0.5

    # Below mastery threshold → explain or ask easy
    if rolling_acc < 0.80:
        action = 0 if len(state["correct_hist"]) % 3 == 0 else 2   # EXPLAIN or ASK_EASY
    else:
        # Mastered → increase difficulty or switch topic
        action = 6 if obs[20] < 0.8 else 8   # INCREASE_DIFF or SWITCH_TOPIC

    # Record simulated correctness from obs (consecutive correct proxy)
    state["correct_hist"].append(1.0 if consec_correct > state.get("prev_cc", 0) else 0.0)
    state["prev_cc"] = consec_correct
    return action


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 4: Zone of Proximal Development (ZPD)
# Concept: always target the difficulty that gives ~70% success rate.
# Implementation: use IRT-style logic on obs to pick appropriate question type.
# Reference: Vygotsky (1978); Csikszentmihalyi's Flow theory
# ─────────────────────────────────────────────────────────────────────────────

def zpd_policy(env, state):
    obs   = env._get_obs()
    know  = float(obs[env.current_topic])   # knowledge of current topic
    diff  = float(obs[20])                  # current difficulty level

    # Estimate probability of success at each difficulty
    # Using simplified IRT: P(correct) ≈ sigmoid(5*(know - difficulty))
    def p_correct(d): return 1/(1+np.exp(-5*(know - d)))

    p_easy   = p_correct(0.25)
    p_medium = p_correct(0.50)
    p_hard   = p_correct(0.80)

    # Pick question closest to 70% target success rate
    target = 0.70
    diffs  = {2: abs(p_easy - target), 3: abs(p_medium - target), 4: abs(p_hard - target)}
    best_action = min(diffs, key=diffs.get)

    # Every 5 steps, explain to consolidate
    step = state.get("step", 0)
    state["step"] = step + 1
    if step % 5 == 0:
        return 0   # EXPLAIN_CONCEPT
    return best_action


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 5: Spaced Repetition (Leitner Box System)
# Concept: topics reviewed more frequently if recently failed;
#          less frequently as they're mastered.
# Reference: Leitner (1972); Ebbinghaus forgetting curve
# ─────────────────────────────────────────────────────────────────────────────

def spaced_repetition_policy(env, state):
    obs = env._get_obs()
    if "boxes" not in state:
        state["boxes"] = [1] * 10        # each topic starts in box 1 (review often)
        state["due"]   = list(range(10)) # all topics due initially
        state["tick"]  = 0

    state["tick"] += 1
    knowledge = obs[0:10]

    # Promote/demote topic in boxes based on knowledge level
    for t in range(10):
        k = knowledge[t]
        if k >= 0.85:   state["boxes"][t] = 5   # mastered → rarely review
        elif k >= 0.65: state["boxes"][t] = 4
        elif k >= 0.45: state["boxes"][t] = 3
        elif k >= 0.25: state["boxes"][t] = 2
        else:           state["boxes"][t] = 1

    # Due topics: review interval = 2^(box-1) steps
    current = env.current_topic
    interval = 2 ** (state["boxes"][current] - 1)

    if state["tick"] % interval == 0:
        return 9   # REVIEW_PREVIOUS
    if knowledge[current] < 0.3:
        return 0   # EXPLAIN if knowledge low
    return 3       # ASK_MEDIUM otherwise


# ─────────────────────────────────────────────────────────────────────────────
# Run all baselines and return results
# ─────────────────────────────────────────────────────────────────────────────

def run_all_baselines(n_episodes: int = 40, seed: int = 7) -> dict:
    print("Running educational baselines...")
    results = {}
    policies = {
        "Random":              random_policy,
        "Fixed Script":        fixed_script_policy,
        "Mastery Learning":    mastery_learning_policy,
        "ZPD (Vygotsky)":      zpd_policy,
        "Spaced Repetition":   spaced_repetition_policy,
    }
    for name, fn in policies.items():
        stats = _run_policy(fn, n_episodes=n_episodes, seed=seed)
        m = compute_metrics(stats)
        results[name] = {"stats": stats, "metrics": m}
        print(f"  {name:<22} reward={m['reward_mean']:5.1f}±{m['reward_sem']:.1f}  "
              f"knowledge={m['knowledge_mean']:.3f}±{m['knowledge_sem']:.3f}  "
              f"mastered={m['mastered_mean']:.1f}")
    return results


if __name__ == "__main__":
    results = run_all_baselines(n_episodes=40)

    print("\n" + "="*72)
    print(f"{'Baseline':<24} {'Reward':<16} {'Knowledge':<16} {'Mastered'}")
    print("-"*72)
    for name, r in results.items():
        m = r["metrics"]
        print(f"{name:<24} {m['reward_mean']:5.1f}±{m['reward_sem']:.1f}      "
              f"{m['knowledge_mean']:.3f}±{m['knowledge_sem']:.3f}      "
              f"{m['mastered_mean']:.1f}")
