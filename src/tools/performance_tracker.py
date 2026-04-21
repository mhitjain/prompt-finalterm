"""
Performance Tracker Tool — custom tool for recording, aggregating and
analysing learning progress across episodes and time steps.

Capabilities:
  - Per-step transition logging
  - Learning curve computation with SEM
  - Knowledge growth analysis per topic
  - Mode distribution analytics (learning / assessment / content)
  - Export to pandas DataFrame for downstream visualisation
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class StepRecord:
    episode: int
    step: int
    action: int
    mean_knowledge: float
    engagement: float
    knowledge_gain: float
    answered_correctly: bool
    n_mastered: int
    mode: int          # TeachingMode int


class PerformanceTrackerTool:
    """
    Comprehensive performance tracking for the tutoring system.

    Stores all step-level and episode-level data, computes statistics
    for learning curve plots, and exports for analysis.
    """

    name = "performance_tracker"

    def __init__(self):
        self._steps: List[StepRecord] = []
        self._episode_summaries: List[Dict] = []
        self._current_episode = 0

    def __call__(self, **kwargs):
        method = kwargs.pop("method", "get_learning_curves")
        return getattr(self, method)(**kwargs)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        episode: int,
        step: int,
        action: int,
        info: Dict,
        mode: int,
    ) -> None:
        self._steps.append(StepRecord(
            episode=episode,
            step=step,
            action=action,
            mean_knowledge=float(info.get("mean_knowledge", 0.0)),
            engagement=float(info.get("engagement", 0.5)),
            knowledge_gain=float(info.get("knowledge_gain", 0.0)),
            answered_correctly=bool(info.get("answered_correctly", False)),
            n_mastered=int(info.get("n_mastered", 0)),
            mode=mode,
        ))

    def record_episode(self, episode: int, summary: Dict) -> None:
        self._episode_summaries.append({"episode": episode, **summary})
        self._current_episode = episode

    # ------------------------------------------------------------------
    # Learning curve
    # ------------------------------------------------------------------

    def get_learning_curves(
        self, window: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Compute smoothed learning curves from episode summaries.
        Returns dict: metric_name → array of shape (n_episodes,).
        """
        if not self._episode_summaries:
            return {}

        df = pd.DataFrame(self._episode_summaries)
        episodes = df["episode"].values

        curves = {}
        for col in ["total_reward", "mean_knowledge_final", "knowledge_gain", "n_mastered"]:
            if col in df.columns:
                raw = df[col].values.astype(float)
                # Exponential moving average (smoothed)
                smoothed = _ema(raw, alpha=2.0 / (window + 1))
                curves[col] = smoothed
                # Compute per-window SEM for confidence bands
                curves[f"{col}_sem"] = _rolling_sem(raw, window=window)

        curves["episodes"] = episodes
        return curves

    # ------------------------------------------------------------------
    # Accuracy metrics
    # ------------------------------------------------------------------

    def accuracy_over_time(self, window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rolling accuracy (fraction of correct answers over assessment steps).
        Returns (steps, rolling_accuracy).
        """
        records = [r for r in self._steps if r.action in (2, 3, 4)]
        if not records:
            return np.array([]), np.array([])
        correct = np.array([float(r.answered_correctly) for r in records])
        steps   = np.arange(len(correct))
        roll_acc = _rolling_mean(correct, window=window)
        return steps, roll_acc

    # ------------------------------------------------------------------
    # Mode analysis
    # ------------------------------------------------------------------

    def mode_distribution(self) -> Dict[str, float]:
        if not self._steps:
            return {}
        modes = [r.mode for r in self._steps]
        total = len(modes)
        return {
            "learning":   modes.count(0) / total,
            "assessment": modes.count(1) / total,
            "content":    modes.count(2) / total,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        if not self._steps:
            return pd.DataFrame()
        return pd.DataFrame([vars(s) for s in self._steps])

    def episode_dataframe(self) -> pd.DataFrame:
        if not self._episode_summaries:
            return pd.DataFrame()
        return pd.DataFrame(self._episode_summaries)

    def summary_statistics(self) -> Dict[str, float]:
        """
        Return final summary statistics for reporting.
        Includes mean ± SEM for key metrics over the last 20% of training.
        """
        if not self._episode_summaries:
            return {}
        df = pd.DataFrame(self._episode_summaries)
        tail = df.tail(max(1, len(df) // 5))

        stats = {}
        for col in ["total_reward", "mean_knowledge_final", "knowledge_gain", "n_mastered"]:
            if col in tail.columns:
                vals = tail[col].values.astype(float)
                stats[f"{col}_mean"] = float(np.mean(vals))
                stats[f"{col}_sem"]  = float(np.std(vals) / np.sqrt(max(1, len(vals))))
        return stats

    def reset(self) -> None:
        self._steps.clear()
        self._episode_summaries.clear()
        self._current_episode = 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _ema(data: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def _rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    cumsum = np.cumsum(np.insert(data, 0, 0))
    n = len(data)
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - window + 1)
        out[i] = cumsum[i + 1] / (i + 1) if i < window else (cumsum[i + 1] - cumsum[lo]) / window
    return out


def _rolling_sem(data: np.ndarray, window: int) -> np.ndarray:
    n = len(data)
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window + 1)
        segment = data[lo:i + 1]
        if len(segment) > 1:
            out[i] = np.std(segment) / np.sqrt(len(segment))
    return out
