"""
Grace — Dewey Framework Algorithms Tutor Agent (RL-enhanced implementation).

Based on Dewey/Chatbots/Grace_A_Chatbot_for_Algorithms.md spec.

Grace's teaching principles:
  - Algorithm classification: identify problem type → select algorithm
  - Complexity visualization: time/space tradeoffs
  - Alternative approaches: greedy vs DP vs divide-and-conquer
  - Interactive quiz mode with immediate feedback
  - Memory: remembers preferred depth, visuals, quiz preferences
"""

from dataclasses import dataclass, field
from typing import Dict, List
from .llm_backend import LLMBackend

GRACE_SYSTEM_PROMPT = """You are Grace, an algorithms tutor in the Dewey educational framework.

Your teaching style:
- Tone: Friendly, inquisitive, knowledgeable
- Humor: "Sometimes solving is easier than sorting! 🧩"
- Methodology: Algorithm classification → Step-by-step breakdown → Complexity analysis → Alternative approaches
- Memory: Remember the student's preferred explanation depth and whether they want visualizations

Core principles:
1. First identify the problem type (sorting, searching, graph, DP, greedy, etc.)
2. Present the algorithm with pseudo-code, not just prose
3. Always discuss time and space complexity using Big-O notation
4. Suggest 1-2 alternative approaches for comparison
5. Use "Watch out for..." to flag common mistakes
6. Ask: "Would you like to see a complexity comparison graph?" (describe it in text)

Topics: Sorting, Searching, Graph Algorithms, Dynamic Programming, Greedy Algorithms,
Divide and Conquer, Data Structures, NP-Completeness, Amortized Analysis.

Keep responses focused. Use code blocks for pseudo-code. Always end with a follow-up question."""

ALGORITHMS_TOPICS = [
    "Sorting Algorithms",
    "Binary Search & Searching",
    "Graph Traversal (BFS/DFS)",
    "Dynamic Programming",
    "Greedy Algorithms",
    "Divide and Conquer",
    "Hash Tables & Hashing",
    "Tree Algorithms",
    "NP-Completeness",
    "Amortized Analysis",
]


class GraceAgent:
    """Grace: RL-enhanced Algorithms Tutor Agent."""

    def __init__(self, backend: LLMBackend):
        self.backend = backend
        self.name = "Grace"
        self.subject = "Algorithms"

    def explain_concept(self, topic: str, depth: int = 2, session_id: str = "default") -> str:
        msg = (
            f"Explain {topic} with pseudo-code and Big-O complexity. "
            f"Depth level: {depth}/3. Structure: Problem type → Algorithm → Pseudo-code → Complexity → Pitfalls."
        )
        return self.backend.generate(GRACE_SYSTEM_PROMPT, msg)

    def show_example(self, topic: str, session_id: str = "default") -> str:
        msg = (
            f"Show a concrete coding example for {topic}. "
            f"Use a realistic interview-style problem. Show the solution step-by-step with complexity analysis."
        )
        return self.backend.generate(GRACE_SYSTEM_PROMPT, msg)

    def ask_question(self, topic: str, difficulty: str = "medium", session_id: str = "default") -> str:
        diff_desc = {
            "easy":   "basic definition or simple application",
            "medium": "standard interview-level problem",
            "hard":   "complex multi-part or proof question",
        }
        msg = (
            f"Generate a {diff_desc.get(difficulty, 'medium')} question about {topic}. "
            f"Don't give the answer. Include what constraints to consider."
        )
        return self.backend.generate(GRACE_SYSTEM_PROMPT, msg)

    def give_hint(self, topic: str, context: str = "", session_id: str = "default") -> str:
        msg = (
            f"Student is stuck on {topic}. Give a hint about which algorithmic technique applies. "
            f"Don't solve it — just redirect their thinking. Context: {context or 'general'}"
        )
        return self.backend.generate(GRACE_SYSTEM_PROMPT, msg)

    def review_prerequisites(self, topic: str, session_id: str = "default") -> str:
        msg = (
            f"Before studying {topic}, what prerequisite data structures or algorithms should the student review? "
            f"Ask one quick check question."
        )
        return self.backend.generate(GRACE_SYSTEM_PROMPT, msg)

    def encourage(self, context: str = "", session_id: str = "default") -> str:
        msg = (
            "Student is struggling with algorithms. Be encouraging and suggest breaking the problem "
            "into smaller subproblems as a strategy. Remind them that pattern recognition comes with practice."
        )
        return self.backend.generate(GRACE_SYSTEM_PROMPT, msg)

    def get_topic_for_action(self, topic_id: int) -> str:
        return ALGORITHMS_TOPICS[min(topic_id, len(ALGORITHMS_TOPICS) - 1)]
