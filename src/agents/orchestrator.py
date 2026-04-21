"""
Orchestrator Agent — coordinates the multi-agent tutoring system.

Architecture
------------
                    ┌─────────────────────────────────────┐
                    │           OrchestratorAgent         │
                    │                                     │
    student obs ───►│  decide_mode()  ──► TutorialAgent  │
                    │                 ──► AssessmentAgent │
                    │                 ──► ContentAgent    │
                    │                                     │
                    │  route_action() ──► TutorialEnv     │
                    └─────────────────────────────────────┘

Decision flow per step
  1. Classify student state into a 'mode' (learning / assessment / review)
  2. Route observation to the responsible specialist agent
  3. Collect action and topic override
  4. Pass transition feedback back to all agents for RL updates
  5. Log and broadcast metrics

Modes
  LEARNING    → TutorialAgent (PPO) selects action
  ASSESSMENT  → AssessmentAgent (bandit) selects question difficulty
  CONTENT     → ContentAgent (bandit) selects presentation style
"""

import os
import numpy as np
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent, AgentMessage
from .tutorial_agent import TutorialAgent
from .assessment_agent import AssessmentAgent
from .content_agent import ContentAgent
from ..environment.tutorial_env import TutorialEnv
from ..tools.knowledge_graph import KnowledgeGraphTool
from ..tools.difficulty_estimator import DifficultyEstimatorTool
from ..tools.performance_tracker import PerformanceTrackerTool


class TeachingMode(IntEnum):
    LEARNING    = 0  # concept delivery / PPO controls
    ASSESSMENT  = 1  # evaluate knowledge / bandit controls difficulty
    CONTENT     = 2  # select presentation style / bandit controls


# Heuristic thresholds
_ASSESS_INTERVAL = 5   # every N steps → run an assessment question
_LOW_KNOWLEDGE   = 0.35  # below this → content mode
_HIGH_KNOWLEDGE  = 0.60  # above this → harder assessment


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator: coordinates TutorialAgent, AssessmentAgent, ContentAgent.

    This is the system controller — it is NOT trained end-to-end but uses
    rule-based heuristics to route control to the appropriate specialist
    agent, enabling modular composition of multiple RL algorithms.
    """

    def __init__(
        self,
        env: TutorialEnv,
        tutorial_agent: TutorialAgent,
        assessment_agent: AssessmentAgent,
        content_agent: ContentAgent,
        agent_id: str = "orchestrator",
        verbose: bool = False,
    ):
        # Build tool set
        kg_tool       = KnowledgeGraphTool()
        diff_tool     = DifficultyEstimatorTool()
        perf_tool     = PerformanceTrackerTool()

        super().__init__(agent_id, tools=[kg_tool, diff_tool, perf_tool])

        self.env              = env
        self.tutorial_agent   = tutorial_agent
        self.assessment_agent = assessment_agent
        self.content_agent    = content_agent
        self.verbose          = verbose

        self._agents: List[BaseAgent] = [
            tutorial_agent, assessment_agent, content_agent
        ]

        self._step_in_episode  = 0
        self._mode_history: List[TeachingMode] = []
        self._episode_stats: List[Dict] = []
        self._total_episodes = 0

        # Performance tracker accumulates cross-episode history
        self._perf_tracker: PerformanceTrackerTool = perf_tool

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        for agent in self._agents:
            agent.reset()
        self._step_in_episode = 0
        self._mode_history = []
        return obs

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def act(self, observation: np.ndarray, context: Dict[str, Any]) -> Tuple[int, Optional[int]]:
        """
        Decide which agent acts and return (action, topic_override).
        Called once per environment step.
        """
        current_topic = self.env.current_topic
        shared_ctx = {"current_topic": current_topic, "step": self._step_in_episode}

        mode = self._decide_mode(observation, current_topic)
        self._mode_history.append(mode)

        if mode == TeachingMode.ASSESSMENT:
            action = self.assessment_agent.act(observation, shared_ctx)
        elif mode == TeachingMode.CONTENT:
            action = self.content_agent.act(observation, shared_ctx)
        else:  # LEARNING mode — PPO
            action = self.tutorial_agent.act(observation, shared_ctx)

        # Determine topic override for SWITCH_TOPIC actions
        topic_override = None
        if action == 8:  # SWITCH_TOPIC
            topic_override = self._select_next_topic(observation)
            if self.verbose:
                print(f"[Orchestrator] SWITCH_TOPIC → topic {topic_override}")

        return action, topic_override

    def run_step(self, obs: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Full orchestrated step:
          1. Select action via routing
          2. Step environment
          3. Distribute feedback to agents
          4. Trigger PPO update if buffer ready
        """
        action, topic_override = self.act(obs, {})
        next_obs, reward, done, info = self.env.step(action, topic_override)

        self._step_in_episode += 1

        # Distribute feedback to all specialist agents
        feedback = {**info, "reward": reward, "done": done}
        for agent in self._agents:
            agent.step_done(feedback)

        # Store transition in tutorial agent's PPO buffer
        self.tutorial_agent.store_transition(reward, done)

        # Trigger PPO update if buffer is full
        self.tutorial_agent.maybe_update(next_obs)

        # Track with performance tool
        self._perf_tracker.record_step(
            episode=self._total_episodes,
            step=self._step_in_episode,
            action=action,
            info=info,
            mode=int(self._mode_history[-1]),
        )

        # Route inter-agent messages
        self._route_messages()

        return next_obs, reward, done, info

    def run_episode(self) -> Dict[str, Any]:
        """Run a complete episode and return summary statistics."""
        obs = self.reset()
        done = False
        while not done:
            obs, _, done, _ = self.run_step(obs)

        summary = self.env.episode_summary()
        summary["mode_distribution"] = self._mode_distribution()
        self._episode_stats.append(summary)
        self._total_episodes += 1

        if self.verbose:
            print(
                f"[Episode {self._total_episodes}] "
                f"reward={summary['total_reward']:.1f}  "
                f"mean_knowledge={summary['mean_knowledge_final']:.3f}  "
                f"mastered={summary['n_mastered']}"
            )
        return summary

    # ------------------------------------------------------------------
    # Mode decision
    # ------------------------------------------------------------------

    def _decide_mode(self, obs: np.ndarray, current_topic: int) -> TeachingMode:
        """
        Heuristic mode selection.
        Every _ASSESS_INTERVAL steps → ASSESSMENT.
        Low knowledge → CONTENT (delivery) mode.
        Otherwise → LEARNING (PPO).
        """
        if self._step_in_episode % _ASSESS_INTERVAL == 0 and self._step_in_episode > 0:
            return TeachingMode.ASSESSMENT

        topic_knowledge = float(obs[current_topic])
        if topic_knowledge < _LOW_KNOWLEDGE:
            return TeachingMode.CONTENT

        return TeachingMode.LEARNING

    # ------------------------------------------------------------------
    # Topic selection
    # ------------------------------------------------------------------

    def _select_next_topic(self, obs: np.ndarray) -> int:
        """
        Use the knowledge graph tool to find the highest-value next topic
        (unlocked, lowest knowledge, most prerequisite-ready).
        """
        knowledge = obs[0:10].tolist()
        try:
            return self.tools["knowledge_graph"].best_next_topic(
                knowledge=knowledge, mastery_threshold=0.85
            )
        except Exception:
            return self.env.select_next_topic()

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def _route_messages(self) -> None:
        """Forward messages from specialist agents to their destinations."""
        for agent in self._agents:
            for msg in agent.flush_outbox():
                if msg.receiver == "orchestrator":
                    self.receive(msg)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _mode_distribution(self) -> Dict[str, float]:
        if not self._mode_history:
            return {}
        total = len(self._mode_history)
        return {
            "learning":   self._mode_history.count(TeachingMode.LEARNING)   / total,
            "assessment": self._mode_history.count(TeachingMode.ASSESSMENT) / total,
            "content":    self._mode_history.count(TeachingMode.CONTENT)    / total,
        }

    def get_stats(self) -> Dict:
        return {
            **super().get_stats(),
            "total_episodes": self._total_episodes,
            "tutorial_stats":   self.tutorial_agent.get_stats(),
            "assessment_stats": self.assessment_agent.get_stats(),
            "content_stats":    self.content_agent.get_stats(),
        }

