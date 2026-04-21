"""Evaluation metrics for the adaptive tutoring system."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EpisodeStats:
    total_reward: float
    mean_knowledge_final: float
    knowledge_gain: float
    n_mastered: int
    steps: int
    disengaged: bool
    profile: int = -1


def compute_metrics(episode_list: List[EpisodeStats]) -> Dict[str, float]:
    """Aggregate episode statistics into mean ± SEM report."""
    if not episode_list:
        return {}

    rewards    = [e.total_reward          for e in episode_list]
    knowledge  = [e.mean_knowledge_final  for e in episode_list]
    k_gain     = [e.knowledge_gain        for e in episode_list]
    mastered   = [e.n_mastered            for e in episode_list]
    dis_rate   = [float(e.disengaged)     for e in episode_list]

    def _ms(arr):
        a = np.array(arr)
        return float(np.mean(a)), float(np.std(a) / max(1, np.sqrt(len(a))))

    r_m, r_s   = _ms(rewards)
    k_m, k_s   = _ms(knowledge)
    kg_m, kg_s = _ms(k_gain)
    ma_m, ma_s = _ms(mastered)
    dr_m, _    = _ms(dis_rate)

    return {
        "reward_mean":    r_m,  "reward_sem":    r_s,
        "knowledge_mean": k_m,  "knowledge_sem": k_s,
        "k_gain_mean":    kg_m, "k_gain_sem":    kg_s,
        "mastered_mean":  ma_m, "mastered_sem":  ma_s,
        "disengagement_rate": dr_m,
        "n_episodes": len(episode_list),
    }


def cohen_d(group_a: List[float], group_b: List[float]) -> float:
    """Cohen's d effect size for comparing two groups."""
    a, b = np.array(group_a), np.array(group_b)
    pooled_std = np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2)
    return float((np.mean(a) - np.mean(b)) / max(pooled_std, 1e-9))


def relative_improvement(baseline: List[float], model: List[float]) -> float:
    """% improvement of model over baseline in mean performance."""
    b_mean = np.mean(baseline)
    m_mean = np.mean(model)
    return float((m_mean - b_mean) / max(abs(b_mean), 1e-9) * 100.0)
