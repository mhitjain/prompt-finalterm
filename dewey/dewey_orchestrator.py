"""
DeweyOrchestrator — connects the RL decision layer to the Dewey LLM agents.

This is the key integration point between our RL system and the Dewey framework.

Architecture
------------
              ┌────────────────────────────────────────────────┐
              │             DeweyOrchestrator                  │
              │                                                │
  student ───►│  RL Layer (PPO + Bandits)                     │
  state        │    TutorialAgent  → WHAT action to take       │
              │    AssessmentAgent → WHAT difficulty           │
              │    ContentAgent   → WHAT style                 │
              │         ↓                                      │
              │  Dewey Agents (LLM-powered)                    │
              │    AdaAgent    → HOW to teach calculus         │
              │    NewtonAgent → HOW to teach physics          │
              │    GraceAgent  → HOW to teach algorithms       │
              │         ↓                                      │
              │  Generated content returned to student        │
              └────────────────────────────────────────────────┘

Key Design Principle
--------------------
  RL decides the PEDAGOGICAL STRATEGY (what action, what difficulty, what style).
  Dewey agents decide the CONTENT (what to actually say/ask).

  This separation means:
  - RL trains fast in simulation (no API calls needed)
  - At deployment, RL policy drives real Dewey agents via LLM
  - The same trained RL policy works regardless of which LLM backend is used
"""

import os
import sys
import numpy as np
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.environment.tutorial_env import TutorialEnv, ACTION_NAMES
from src.agents.orchestrator import OrchestratorAgent
from src.agents.tutorial_agent import TutorialAgent
from src.agents.assessment_agent import AssessmentAgent
from src.agents.content_agent import ContentAgent
from src.environment.student_simulator import TOPIC_NAMES

from .llm_backend import LLMBackend
from .ada_agent import AdaAgent
from .newton_agent import NewtonAgent
from .grace_agent import GraceAgent


# Map TutorialEnv action index → Dewey agent method name
ACTION_TO_METHOD = {
    0:  "explain_concept",
    1:  "show_example",
    2:  "ask_question_easy",
    3:  "ask_question_medium",
    4:  "ask_question_hard",
    5:  "give_hint",
    6:  None,   # INCREASE_DIFFICULTY — meta-action, no content needed
    7:  None,   # DECREASE_DIFFICULTY
    8:  None,   # SWITCH_TOPIC — handled by topic selection
    9:  "review_prerequisites",
    10: "encourage",
}

# Map subject domain → which Dewey agent to use
SUBJECT_AGENT_MAP = {
    "calculus":  "ada",
    "physics":   "newton",
    "algorithms": "grace",
    "default":   "ada",
}


class DeweyOrchestrator:
    """
    Full Dewey + RL integration.

    In TRAINING mode: uses TutorialEnv student simulator (fast, no API)
    In DEMO mode: runs RL policy and calls real Dewey LLM agents for content

    Parameters
    ----------
    env         : TutorialEnv (the student simulation environment)
    rl_system   : OrchestratorAgent (the RL orchestrator)
    backend     : LLMBackend (Claude API or simulation)
    subject     : "calculus" | "physics" | "algorithms"
    """

    def __init__(
        self,
        env: TutorialEnv,
        rl_system: OrchestratorAgent,
        backend: LLMBackend,
        subject: str = "calculus",
        verbose: bool = True,
    ):
        self.env       = env
        self.rl_system = rl_system
        self.verbose   = verbose
        self.subject   = subject

        # Initialise Dewey agents with shared backend
        self.ada    = AdaAgent(backend)
        self.newton = NewtonAgent(backend)
        self.grace  = GraceAgent(backend)

        self._active_agent = self._pick_agent(subject)
        self._session_id = "student_001"
        self._step_log: list = []

    # ------------------------------------------------------------------
    # Agent selection
    # ------------------------------------------------------------------

    def _pick_agent(self, subject: str):
        mapping = {"calculus": self.ada, "physics": self.newton, "algorithms": self.grace}
        return mapping.get(subject.lower(), self.ada)

    # ------------------------------------------------------------------
    # Demo: single interactive session
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self._obs = self.rl_system.reset()
        return self._obs

    def step(self) -> Dict:
        """
        Take one tutoring step.
        Called by the API server for each student interaction.
        Returns a dict with action, reward, content, done, and state info.
        """
        if not hasattr(self, "_obs") or self._obs is None:
            self._obs = self.rl_system.reset()

        current_topic_id = self.env.current_topic
        topic_name = TOPIC_NAMES[current_topic_id]

        action, topic_override = self.rl_system.act(self._obs, {})
        content = self._dispatch_to_dewey(action, topic_name, topic_override)
        self._obs, reward, done, info = self.env.step(action, topic_override)
        self.rl_system.tutorial_agent.store_transition(reward, done)

        return {
            "action": action,
            "action_name": ACTION_NAMES[action],
            "topic": topic_name,
            "content": content or "",
            "reward": float(reward),
            "done": bool(done),
            "mean_knowledge": float(info.get("mean_knowledge", 0)),
            "engagement": float(info.get("engagement", 0)),
            "n_mastered": int(info.get("n_mastered", 0)),
        }

    def run_demo_session(self, n_steps: int = 10) -> None:
        """
        Run a demo tutoring session:
          - RL policy selects actions
          - Dewey agent generates real educational content
          - Prints a readable transcript
        """
        print("\n" + "=" * 70)
        print(f"  DEWEY ADAPTIVE TUTORING SESSION — {self._active_agent.name} ({self._active_agent.subject})")
        print(f"  RL Policy: PPO + Thompson Sampling")
        print("=" * 70 + "\n")

        obs = self.rl_system.reset()
        done = False
        step = 0

        while not done and step < n_steps:
            step += 1
            current_topic_id = self.env.current_topic
            topic_name = TOPIC_NAMES[current_topic_id]

            # RL selects action
            action, topic_override = self.rl_system.act(obs, {})
            action_name = ACTION_NAMES[action]

            # Generate Dewey content for this action
            content = self._dispatch_to_dewey(action, topic_name, topic_override)

            # Step environment
            obs, reward, done, info = self.env.step(action, topic_override)
            self.rl_system.tutorial_agent.store_transition(reward, done)

            # Log and print
            self._step_log.append({
                "step": step, "action": action_name,
                "topic": topic_name, "content": content,
                "reward": reward, "knowledge": info["mean_knowledge"],
                "engagement": info["engagement"],
            })

            print(f"Step {step:2d} | Topic: {topic_name:<22} | RL Action: {action_name}")
            print(f"       Engagement: {info['engagement']:.2f}  "
                  f"Knowledge: {info['mean_knowledge']:.3f}  "
                  f"Reward: {reward:+.2f}")
            if content:
                print(f"\n  [{self._active_agent.name}]: {content[:200]}{'...' if len(content)>200 else ''}")
            print()

            if done:
                print(f"\n{'─'*70}")
                summary = self.env.episode_summary()
                print(f"Session complete! Knowledge gained: {summary['knowledge_gain']:.3f}")
                print(f"Topics mastered: {summary['n_mastered']}/10")
                print(f"{'─'*70}\n")

    # ------------------------------------------------------------------
    # Content dispatch
    # ------------------------------------------------------------------

    def _dispatch_to_dewey(
        self, action: int, topic_name: str, topic_override: Optional[int]
    ) -> Optional[str]:
        """
        Map RL action → Dewey agent method → LLM-generated content.
        Returns None for meta-actions (difficulty adjustment) that need no content.
        """
        agent = self._active_agent

        if action == 0:
            return agent.explain_concept(topic_name, session_id=self._session_id)
        elif action == 1:
            return agent.show_example(topic_name, session_id=self._session_id)
        elif action == 2:
            return agent.ask_question(topic_name, difficulty="easy", session_id=self._session_id)
        elif action == 3:
            return agent.ask_question(topic_name, difficulty="medium", session_id=self._session_id)
        elif action == 4:
            return agent.ask_question(topic_name, difficulty="hard", session_id=self._session_id)
        elif action == 5:
            return agent.give_hint(topic_name, session_id=self._session_id)
        elif action == 6:
            return "📈 [Difficulty increased — moving to more challenging material]"
        elif action == 7:
            return "📉 [Difficulty decreased — let's consolidate before moving forward]"
        elif action == 8:
            new_topic = TOPIC_NAMES[topic_override] if topic_override is not None else topic_name
            return f"🔄 [Switching focus to: {new_topic}]"
        elif action == 9:
            return agent.review_prerequisites(topic_name, session_id=self._session_id)
        elif action == 10:
            return agent.encourage(session_id=self._session_id)
        return None

    # ------------------------------------------------------------------
    # Training loop (uses simulator, no LLM calls)
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int, verbose: bool = False) -> Dict:
        """
        Train the RL policy in simulation mode (no LLM calls).
        Identical to the standard training loop — Dewey agents are not called.
        """
        obs = self.rl_system.reset()
        episode = 0
        episode_rewards = []

        for t in range(total_timesteps):
            obs, reward, done, info = self.rl_system.run_step(obs)
            if done:
                summary = self.env.episode_summary()
                episode_rewards.append(summary["total_reward"])
                episode += 1
                obs = self.rl_system.reset()
                if verbose and episode % 100 == 0:
                    print(
                        f"  [Train] ep={episode}  t={t+1}  "
                        f"reward={np.mean(episode_rewards[-20:]):.1f}  "
                        f"knowledge={summary['mean_knowledge_final']:.3f}"
                    )

        return {
            "episodes": episode,
            "mean_reward_last20": float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0,
        }

    # ------------------------------------------------------------------
    # Transcript export
    # ------------------------------------------------------------------

    def get_transcript(self) -> str:
        """Return formatted session transcript for the report."""
        lines = [f"DEWEY SESSION TRANSCRIPT — {self._active_agent.name}", "=" * 60]
        for entry in self._step_log:
            lines.append(
                f"\nStep {entry['step']} | {entry['action']} on '{entry['topic']}'"
            )
            if entry["content"]:
                lines.append(f"  → {entry['content'][:300]}")
            lines.append(
                f"  [knowledge={entry['knowledge']:.3f}  engagement={entry['engagement']:.2f}  "
                f"reward={entry['reward']:+.2f}]"
            )
        return "\n".join(lines)
