"""
Dewey RL Demo — shows the trained RL policy driving real Dewey-style agents.

Usage:
    # Simulation mode (no API key needed):
    python dewey_demo.py --subject calculus --steps 12

    # With OpenAI GPT key:
    python dewey_demo.py --subject calculus --steps 12 --openai_key sk-...

    # With env variable:
    OPENAI_API_KEY=sk-... python dewey_demo.py --subject calculus --steps 12

    # Physics or Algorithms:
    python dewey_demo.py --subject physics --steps 10
    python dewey_demo.py --subject algorithms --steps 10

    # Train first, then demo:
    python dewey_demo.py --train --timesteps 80000 --subject calculus --openai_key sk-...
"""

import argparse
import os
import sys
import numpy as np
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from src.environment.tutorial_env import TutorialEnv
from src.agents.tutorial_agent import TutorialAgent
from src.agents.assessment_agent import AssessmentAgent
from src.agents.content_agent import ContentAgent
from src.agents.orchestrator import OrchestratorAgent
from src.rl.ppo import PPOConfig
from dewey.llm_backend import LLMBackend
from dewey.dewey_orchestrator import DeweyOrchestrator


def parse_args():
    p = argparse.ArgumentParser(description="Dewey RL Adaptive Tutoring Demo")
    p.add_argument("--subject",    choices=["calculus", "physics", "algorithms"], default="calculus")
    p.add_argument("--steps",      type=int, default=12, help="Steps in the demo session")
    p.add_argument("--train",      action="store_true", help="Train RL policy before demo")
    p.add_argument("--timesteps",  type=int, default=60000, help="Training timesteps")
    p.add_argument("--checkpoint", type=str, default=None, help="Load saved checkpoint")
    p.add_argument("--openai_key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    p.add_argument("--api_key",    type=str, default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def build_rl_system(seed: int = 42) -> tuple[TutorialEnv, OrchestratorAgent]:
    env = TutorialEnv(n_topics=10, max_steps=50, seed=seed)
    cfg = PPOConfig(lr=3e-4, hidden_dim=256, update_interval=1024)
    tutorial   = TutorialAgent(ppo_config=cfg)
    assessment = AssessmentAgent(alpha=1.0)
    content    = ContentAgent(alpha=1.0)
    orchestrator = OrchestratorAgent(env, tutorial, assessment, content, verbose=False)
    return env, orchestrator


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("\n" + "█" * 70)
    print("  DEWEY FRAMEWORK — RL-ENHANCED ADAPTIVE TUTORING SYSTEM")
    print("  PPO Policy Gradient + Linear Thompson Sampling")
    print("█" * 70)

    # Build RL system
    env, rl_system = build_rl_system(args.seed)

    # Build LLM backend — OpenAI takes priority if provided
    backend = LLMBackend(
        openai_api_key=args.openai_key or os.environ.get("OPENAI_API_KEY", "") or None,
        api_key=args.api_key or os.environ.get("ANTHROPIC_API_KEY", "") or None,
    )

    # Build Dewey orchestrator
    dewey = DeweyOrchestrator(
        env=env,
        rl_system=rl_system,
        backend=backend,
        subject=args.subject,
        verbose=True,
    )

    # --- Optional training phase ---
    if args.train:
        print(f"\n[Phase 1] Training RL policy for {args.timesteps:,} timesteps...")
        result = dewey.train(total_timesteps=args.timesteps, verbose=True)
        print(f"  Training complete. Mean reward (last 20 eps): {result['mean_reward_last20']:.1f}")

        ckpt_path = f"results/ppo_checkpoints/dewey_{args.subject}_trained.pt"
        os.makedirs("results/ppo_checkpoints", exist_ok=True)
        rl_system.tutorial_agent.save(ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    # --- Load checkpoint if provided ---
    if args.checkpoint and os.path.exists(args.checkpoint):
        rl_system.tutorial_agent.load(args.checkpoint)
        print(f"\n  Loaded checkpoint: {args.checkpoint}")

    # --- Demo session ---
    print(f"\n[Phase 2] Running Dewey demo session ({args.steps} steps, subject={args.subject})")
    print(f"  LLM backend: {backend.provider_name}\n")

    dewey.run_demo_session(n_steps=args.steps)

    # --- Print transcript ---
    transcript = dewey.get_transcript()
    transcript_path = f"results/dewey_{args.subject}_transcript.txt"
    os.makedirs("results", exist_ok=True)
    with open(transcript_path, "w") as f:
        f.write(transcript)
    print(f"\nFull transcript saved: {transcript_path}")

    print("\n" + "█" * 70)
    print("  HOW THIS WORKS:")
    print("  ┌─ PPO policy → decides WHAT to do (explain? ask? hint? switch topic?)")
    print("  ├─ Thompson Sampling → decides question difficulty & content style")
    print(f"  └─ {dewey._active_agent.name} ({dewey._active_agent.subject}) → generates the actual content")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()
