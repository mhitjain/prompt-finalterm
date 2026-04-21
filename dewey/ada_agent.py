"""
Ada — Dewey Framework Calculus Tutor Agent (RL-enhanced implementation).

Based on the Dewey/Chatbots/Ada_A_Chatbot_for_Introductory_Calculus.md spec.

Ada's teaching principles (from the Dewey spec):
  - Socratic questioning: hints before answers
  - Progress-aware: respects syllabus scope
  - Structured explanations: Definition → Formula → Example → Visual
  - Exam-timing awareness
  - Adaptive depth control

RL Enhancement:
  The RL layer (PPO + bandits) decides WHAT Ada does each step.
  Ada handles HOW — generating the actual educational content.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .llm_backend import LLMBackend

# Ada's system prompt — directly from the Dewey specification
ADA_SYSTEM_PROMPT = """You are Ada, a friendly and knowledgeable calculus tutor in the Dewey educational framework.

Your teaching style:
- Tone: Casual, mentor-like, encouraging
- Style: Friendly, college-level, concise
- Methodology: Socratic — guide students to answers through hints and questions rather than direct answers
- Structure: Definition → Explanation → Formula → Example → Visual description

Core principles:
1. Ask students about their current progress before diving in
2. Limit scope to what the student has studied so far
3. Provide step-by-step breakdowns when requested
4. Offer disclaimers for topics not yet covered
5. Before exams: prioritise review and practice problems
6. Always ask: "Would you like the direct answer or a full explanation?"

Topics you cover: Limits, Derivatives, Integrals, Series, Multivariable Calculus, Differential Equations.
Always be encouraging. Students learn calculus best through guided discovery, not passive reading.
Keep responses concise (3-5 sentences unless a worked example is requested)."""

CALCULUS_TOPICS = [
    "Limits and Continuity",
    "Derivatives and Differentiation Rules",
    "Applications of Derivatives",
    "Integration Techniques",
    "Fundamental Theorem of Calculus",
    "Series and Sequences",
    "Multivariable Calculus",
    "Differential Equations",
    "Vector Calculus",
    "Optimization",
]


@dataclass
class AdaSession:
    """Tracks a student's session state with Ada."""
    student_id: str
    chapter_progress: int = 1
    preferred_depth: int = 2        # 1=simple, 2=standard, 3=detailed
    wants_direct_answers: bool = False
    interaction_history: List[Dict] = field(default_factory=list)
    correct_count: int = 0
    incorrect_count: int = 0


class AdaAgent:
    """
    Ada: RL-enhanced Calculus Tutor Agent.

    The RL system (PPO + bandits) calls Ada's methods to generate content.
    Ada wraps the Dewey spec into executable, LLM-powered teaching actions.
    """

    def __init__(self, backend: LLMBackend):
        self.backend = backend
        self.name = "Ada"
        self.subject = "Calculus"
        self._sessions: Dict[str, AdaSession] = {}

    def get_or_create_session(self, student_id: str) -> AdaSession:
        if student_id not in self._sessions:
            self._sessions[student_id] = AdaSession(student_id=student_id)
        return self._sessions[student_id]

    # ------------------------------------------------------------------
    # RL-callable teaching actions
    # All methods correspond to TutorialEnv actions
    # ------------------------------------------------------------------

    def explain_concept(self, topic: str, depth: int = 2, session_id: str = "default") -> str:
        """ACTION 0: Explain a calculus concept. Depth 1=simple, 2=standard, 3=detailed."""
        session = self.get_or_create_session(session_id)
        depth_desc = {1: "simple, beginner-friendly", 2: "standard college-level", 3: "rigorous and detailed"}
        msg = (
            f"Explain {topic} in calculus at a {depth_desc.get(depth, 'standard')} level. "
            f"Follow the Dewey structure: definition, key formula, intuitive explanation. "
            f"Be concise. Ask a follow-up question at the end."
        )
        response = self.backend.generate(ADA_SYSTEM_PROMPT, msg)
        self._log(session, "explain", topic, response)
        return response

    def show_example(self, topic: str, session_id: str = "default") -> str:
        """ACTION 1: Show a worked example."""
        session = self.get_or_create_session(session_id)
        msg = (
            f"Show a concrete worked example for {topic} in calculus. "
            f"Use the format: Problem → Step-by-step solution → Key insight. "
            f"Make it something a college student would encounter on a midterm."
        )
        response = self.backend.generate(ADA_SYSTEM_PROMPT, msg)
        self._log(session, "example", topic, response)
        return response

    def ask_question(self, topic: str, difficulty: str = "medium", session_id: str = "default") -> str:
        """ACTIONS 2/3/4: Generate a question at easy/medium/hard difficulty."""
        session = self.get_or_create_session(session_id)
        diff_map = {
            "easy":   "conceptual, definition-level, straightforward",
            "medium": "application-level, requires using a formula or technique",
            "hard":   "proof-level or multi-step, requires deep understanding",
        }
        msg = (
            f"Generate a single {diff_map.get(difficulty, 'medium')} calculus question about {topic}. "
            f"Do NOT give the answer yet. Ask the student to attempt it first. "
            f"Format: [Question] then [Hint available if needed]."
        )
        response = self.backend.generate(ADA_SYSTEM_PROMPT, msg)
        self._log(session, f"question_{difficulty}", topic, response)
        return response

    def give_hint(self, topic: str, context: str = "", session_id: str = "default") -> str:
        """ACTION 5: Give a Socratic hint without revealing the answer."""
        session = self.get_or_create_session(session_id)
        msg = (
            f"The student is stuck on {topic} in calculus. "
            f"Give a Socratic hint — guide them toward the answer without revealing it. "
            f"Use a leading question or point to the most relevant concept. "
            f"Be encouraging. Context: {context or 'general difficulty'}"
        )
        response = self.backend.generate(ADA_SYSTEM_PROMPT, msg)
        self._log(session, "hint", topic, response)
        return response

    def review_prerequisites(self, topic: str, session_id: str = "default") -> str:
        """ACTION 9: Review prerequisite material using spaced repetition."""
        session = self.get_or_create_session(session_id)
        msg = (
            f"The student is about to study {topic} in calculus. "
            f"Briefly review the key prerequisite concepts they need to know first. "
            f"Ask them 1 quick check question to confirm they're ready to proceed."
        )
        response = self.backend.generate(ADA_SYSTEM_PROMPT, msg)
        self._log(session, "review", topic, response)
        return response

    def encourage(self, context: str = "", session_id: str = "default") -> str:
        """ACTION 10: Provide motivational support."""
        session = self.get_or_create_session(session_id)
        msg = (
            f"The student is struggling and needs encouragement. "
            f"Be warm, genuine, and remind them that calculus takes time to click. "
            f"Suggest a concrete next step to rebuild confidence. "
            f"Context: {context or 'general struggle'}"
        )
        response = self.backend.generate(ADA_SYSTEM_PROMPT, msg)
        self._log(session, "encourage", "", response)
        return response

    def evaluate_response(self, topic: str, student_answer: str, session_id: str = "default") -> Dict:
        """Evaluate a student's answer and return feedback + score."""
        session = self.get_or_create_session(session_id)
        msg = (
            f"A student answered a calculus question about {topic}. "
            f"Their answer: '{student_answer}'. "
            f"Evaluate: is it correct? Provide brief feedback. "
            f"Format your response as: CORRECT/INCORRECT: [one-line verdict]. [2-3 sentence feedback]"
        )
        response = self.backend.generate(ADA_SYSTEM_PROMPT, msg)
        correct = response.upper().startswith("CORRECT")
        if correct:
            session.correct_count += 1
        else:
            session.incorrect_count += 1
        return {"correct": correct, "feedback": response, "topic": topic}

    # ------------------------------------------------------------------
    # Session utilities
    # ------------------------------------------------------------------

    def _log(self, session: AdaSession, action: str, topic: str, response: str) -> None:
        session.interaction_history.append({
            "action": action, "topic": topic, "response_preview": response[:80]
        })

    def get_topic_for_action(self, topic_id: int) -> str:
        return CALCULUS_TOPICS[min(topic_id, len(CALCULUS_TOPICS) - 1)]

    def session_summary(self, session_id: str) -> Dict:
        session = self._sessions.get(session_id)
        if not session:
            return {}
        total = session.correct_count + session.incorrect_count
        return {
            "student_id":    session.student_id,
            "interactions":  len(session.interaction_history),
            "accuracy":      session.correct_count / max(1, total),
            "correct":       session.correct_count,
            "incorrect":     session.incorrect_count,
        }
