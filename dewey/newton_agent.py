"""
Newton — Dewey Framework Physics Tutor Agent (RL-enhanced implementation).

Based on Dewey/Chatbots/Newton_A_Chatbot_for_Introductory_Physics.md spec.

Newton's teaching principles:
  - Automatic visualization: "Use Matplotlib everywhere you can without asking"
  - Problem breakdown: identify phases and parameters before solving
  - Real-world analogies: connect physics to everyday scenarios
  - Forced engagement: always ask follow-up questions
  - Humor: "Gravity — it's pulling us together! 🌍"
"""

from typing import Dict
from .llm_backend import LLMBackend

NEWTON_SYSTEM_PROMPT = """You are Newton, a physics tutor in the Dewey educational framework.

Your teaching style:
- Tone: Friendly, casual, academic — like a grad student who loves physics
- Humor: Use physics puns naturally ("Gravity — it's pulling us together!")
- Accessibility: Simplified language by default, dive deeper on request
- Visualization: Always suggest/describe a diagram or graph where relevant

Core principles:
1. Before solving: identify all given variables and what to find
2. Break into phases (for motion problems: identify intervals, forces, constraints)
3. Use real-world analogies before abstract formulas
4. Always ask: "Does this make physical sense?" after a solution
5. Generate Matplotlib graph descriptions (text) for any motion or field visualization
6. Always ask a follow-up question or give a brainstorming prompt

Topics: Classical Mechanics, Kinematics, Forces, Energy, Momentum,
Waves, Thermodynamics, Electromagnetism, Modern Physics.

Keep answers focused. Use numbered steps for problem solutions.
End with a follow-up question like 'What happens if we double the mass?'"""

PHYSICS_TOPICS = [
    "Kinematics (1D Motion)",
    "Newton's Laws of Motion",
    "Work, Energy & Power",
    "Momentum & Collisions",
    "Rotational Motion",
    "Oscillations & Waves",
    "Thermodynamics",
    "Electrostatics",
    "Magnetism & Induction",
    "Modern Physics",
]


class NewtonAgent:
    """Newton: RL-enhanced Physics Tutor Agent."""

    def __init__(self, backend: LLMBackend):
        self.backend = backend
        self.name = "Newton"
        self.subject = "Physics"

    def explain_concept(self, topic: str, depth: int = 2, session_id: str = "default") -> str:
        msg = (
            f"Explain {topic} in physics. Start with a real-world analogy, then the formula, "
            f"then a brief example. Depth: {depth}/3. Describe a helpful diagram in text."
        )
        return self.backend.generate(NEWTON_SYSTEM_PROMPT, msg)

    def show_example(self, topic: str, session_id: str = "default") -> str:
        msg = (
            f"Solve a typical {topic} problem step by step. "
            f"Format: Given → Find → Diagram description → Solution → Physical sense check."
        )
        return self.backend.generate(NEWTON_SYSTEM_PROMPT, msg)

    def ask_question(self, topic: str, difficulty: str = "medium", session_id: str = "default") -> str:
        diff_desc = {
            "easy":   "conceptual, no calculation needed",
            "medium": "single-step calculation with real numbers",
            "hard":   "multi-step problem combining multiple concepts",
        }
        msg = (
            f"Generate a {diff_desc.get(difficulty, 'medium')} physics question about {topic}. "
            f"Include a diagram description. Don't give the answer yet."
        )
        return self.backend.generate(NEWTON_SYSTEM_PROMPT, msg)

    def give_hint(self, topic: str, context: str = "", session_id: str = "default") -> str:
        msg = (
            f"Student is stuck on a {topic} problem. Give a Socratic hint: "
            f"point to the most relevant law or formula without solving it. "
            f"Ask: 'What forces are acting here?' or similar. Context: {context or 'general'}"
        )
        return self.backend.generate(NEWTON_SYSTEM_PROMPT, msg)

    def review_prerequisites(self, topic: str, session_id: str = "default") -> str:
        msg = (
            f"Student is about to study {topic}. Briefly review what they need to know first. "
            f"Quick check question to confirm readiness."
        )
        return self.backend.generate(NEWTON_SYSTEM_PROMPT, msg)

    def encourage(self, context: str = "", session_id: str = "default") -> str:
        msg = (
            "Student is struggling. Be encouraging with a physics humour touch. "
            "Remind them physics builds intuition over time. Suggest a simpler approach."
        )
        return self.backend.generate(NEWTON_SYSTEM_PROMPT, msg)

    def get_topic_for_action(self, topic_id: int) -> str:
        return PHYSICS_TOPICS[min(topic_id, len(PHYSICS_TOPICS) - 1)]
