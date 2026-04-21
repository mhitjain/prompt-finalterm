"""
Visualisation utilities for learning curves, agent behaviour, and analysis plots.
All figures are publication-quality (300 dpi, clean style).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional, Tuple

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = sns.color_palette("muted", 8)


class Visualizer:
    """Centralised plotting interface."""

    def __init__(self, save_dir: str = "results/figures"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Learning curves
    # ------------------------------------------------------------------

    def plot_learning_curves(
        self,
        curves_dict: Dict[str, Dict],
        metric: str = "mean_knowledge_final",
        title: str = "Learning Curves",
        filename: str = "learning_curves.png",
    ) -> str:
        """
        Plot smoothed learning curves with SEM confidence bands.

        Parameters
        ----------
        curves_dict : {label: curves} where curves is from PerformanceTracker.get_learning_curves()
        metric      : which metric to plot
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        for i, (label, curves) in enumerate(curves_dict.items()):
            if metric not in curves:
                continue
            eps = curves.get("episodes", np.arange(len(curves[metric])))
            y   = curves[metric]
            sem = curves.get(f"{metric}_sem", np.zeros_like(y))
            color = COLORS[i % len(COLORS)]
            ax.plot(eps, y, label=label, color=color, linewidth=2)
            ax.fill_between(eps, y - sem, y + sem, alpha=0.2, color=color)

        ax.set_xlabel("Training Episode", fontsize=12)
        metric_labels = {
            "mean_knowledge_final": "Mean Knowledge (0–1)",
            "total_reward":         "Episode Reward",
            "knowledge_gain":       "Knowledge Gain per Episode",
            "n_mastered":           "Topics Mastered",
        }
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_xlim(left=0)
        plt.tight_layout()
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 2. Reward comparison bar chart
    # ------------------------------------------------------------------

    def plot_comparison_bar(
        self,
        results: Dict[str, Tuple[float, float]],  # {label: (mean, sem)}
        metric: str = "Mean Episode Reward",
        filename: str = "comparison_bar.png",
    ) -> str:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(results.keys())
        means  = [results[l][0] for l in labels]
        sems   = [results[l][1] for l in labels]

        bars = ax.bar(labels, means, color=COLORS[:len(labels)], width=0.5, zorder=3)
        ax.errorbar(
            labels, means, yerr=[1.96 * s for s in sems],
            fmt="none", color="black", capsize=6, linewidth=1.5, zorder=4,
        )
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"Algorithm Comparison — {metric}", fontsize=14, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", zorder=0)

        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(sems) * 0.5,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=10,
            )
        plt.tight_layout()
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 3. Knowledge heatmap (per topic, per profile)
    # ------------------------------------------------------------------

    def plot_knowledge_heatmap(
        self,
        profile_knowledge: Dict[str, np.ndarray],
        topic_names: List[str],
        filename: str = "knowledge_heatmap.png",
    ) -> str:
        profiles = list(profile_knowledge.keys())
        data = np.array([profile_knowledge[p] for p in profiles])

        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(
            data, ax=ax,
            xticklabels=topic_names, yticklabels=profiles,
            cmap="YlOrRd", vmin=0, vmax=1, annot=True, fmt=".2f",
            linewidths=0.5, cbar_kws={"label": "Knowledge Level"},
        )
        ax.set_title("Final Knowledge per Topic × Student Profile", fontsize=13, fontweight="bold")
        ax.set_xlabel("Topic", fontsize=11)
        ax.set_ylabel("Profile", fontsize=11)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 4. Bandit arm selection over time
    # ------------------------------------------------------------------

    def plot_bandit_arm_distribution(
        self,
        arm_counts: Dict[str, int],
        title: str = "Bandit Arm Selection Distribution",
        filename: str = "bandit_arms.png",
    ) -> str:
        arms  = list(arm_counts.keys())
        counts = list(arm_counts.values())
        total  = max(sum(counts), 1)
        fracs  = [c / total for c in counts]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(arms, fracs, color=COLORS[:len(arms)], zorder=3)
        ax.set_ylabel("Fraction of Selections", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", zorder=0)
        for bar, frac in zip(bars, fracs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{frac:.1%}",
                ha="center", va="bottom", fontsize=10,
            )
        plt.tight_layout()
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 5. PPO training diagnostics
    # ------------------------------------------------------------------

    def plot_ppo_diagnostics(
        self,
        metrics_history: List[Dict],
        filename: str = "ppo_diagnostics.png",
    ) -> str:
        if not metrics_history:
            return ""
        keys = ["policy_loss", "value_loss", "entropy", "clip_fraction"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for ax, key in zip(axes, keys):
            vals = [m[key] for m in metrics_history if key in m]
            updates = np.arange(len(vals))
            ax.plot(updates, vals, color=COLORS[0], linewidth=1.5)
            ax.set_title(key.replace("_", " ").title(), fontsize=11)
            ax.set_xlabel("PPO Update")
        plt.suptitle("PPO Training Diagnostics", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 6. Mode distribution pie chart
    # ------------------------------------------------------------------

    def plot_mode_distribution(
        self,
        mode_dist: Dict[str, float],
        filename: str = "mode_distribution.png",
    ) -> str:
        labels = list(mode_dist.keys())
        sizes  = list(mode_dist.values())
        colors = [COLORS[0], COLORS[1], COLORS[2]]

        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors[:len(labels)],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for t in autotexts:
            t.set_fontsize(11)
        ax.set_title("Teaching Mode Distribution", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path
