"""
LLM Backend — supports Groq (llama3), OpenAI GPT, Anthropic Claude,
and a template-based simulation fallback (no API key needed for training).

Auto-detection priority:
  1. GROQ_API_KEY       → uses llama-3.1-8b-instant  (fast, free tier)
  2. OPENAI_API_KEY     → uses gpt-4o-mini
  3. ANTHROPIC_API_KEY  → uses claude-haiku
  4. No key found       → simulation mode

Usage:
    from dotenv import load_dotenv
    load_dotenv()
    backend = LLMBackend()          # auto-detects from env
"""

import os
import random
from typing import Optional


class LLMBackend:

    GROQ_MODEL    = "llama-3.1-8b-instant"  # fast, free tier
    OPENAI_MODEL  = "gpt-4o-mini"
    CLAUDE_MODEL  = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        mode: str = "auto",
        groq_api_key: str = None,
        openai_api_key: str = None,
        api_key: str = None,       # Anthropic key
    ):
        self._groq_client   = None
        self._openai_client = None
        self._claude_client = None
        self.mode = mode

        if mode == "auto":
            groq_key = groq_api_key  or os.environ.get("GROQ_API_KEY",       "")
            oai_key  = openai_api_key or os.environ.get("OPENAI_API_KEY",    "")
            ant_key  = api_key        or os.environ.get("ANTHROPIC_API_KEY", "")
            if groq_key:
                mode = "groq"
            elif oai_key:
                mode = "openai"
            elif ant_key:
                mode = "claude"
            else:
                mode = "sim"
            self.mode = mode

        if mode == "groq":
            self._init_groq(groq_api_key or os.environ.get("GROQ_API_KEY"))
        elif mode == "openai":
            self._init_openai(openai_api_key or os.environ.get("OPENAI_API_KEY"))
        elif mode == "claude":
            self._init_claude(api_key or os.environ.get("ANTHROPIC_API_KEY"))
        else:
            print("[LLMBackend] Simulation mode (no API key required)")

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _init_groq(self, key: str) -> None:
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=key)
            print(f"[LLMBackend] Using Groq ({self.GROQ_MODEL})")
        except ImportError:
            print("[LLMBackend] groq package not installed — pip install groq")
            self.mode = "sim"
        except Exception as e:
            print(f"[LLMBackend] Groq init failed: {e} — falling back to simulation")
            self.mode = "sim"

    def _init_openai(self, key: str) -> None:
        try:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=key)
            print(f"[LLMBackend] Using OpenAI ({self.OPENAI_MODEL})")
        except ImportError:
            print("[LLMBackend] openai package not installed — falling back to simulation")
            self.mode = "sim"
        except Exception as e:
            print(f"[LLMBackend] OpenAI init failed: {e} — falling back to simulation")
            self.mode = "sim"

    def _init_claude(self, key: str) -> None:
        try:
            import anthropic
            self._claude_client = anthropic.Anthropic(api_key=key)
            print(f"[LLMBackend] Using Anthropic Claude ({self.CLAUDE_MODEL})")
        except ImportError:
            print("[LLMBackend] anthropic package not installed — falling back to simulation")
            self.mode = "sim"
        except Exception as e:
            print(f"[LLMBackend] Claude init failed: {e} — falling back to simulation")
            self.mode = "sim"

    # ── Public interface ──────────────────────────────────────────────────────

    def generate(self, system_prompt: str, user_message: str, max_tokens: int = 120) -> str:
        if self.mode == "groq"   and self._groq_client:
            return self._call_groq(system_prompt, user_message, max_tokens)
        if self.mode == "openai" and self._openai_client:
            return self._call_openai(system_prompt, user_message, max_tokens)
        if self.mode == "claude" and self._claude_client:
            return self._call_claude(system_prompt, user_message, max_tokens)
        return self._simulate(system_prompt, user_message)

    @property
    def provider_name(self) -> str:
        return {
            "groq":   f"Groq ({self.GROQ_MODEL})",
            "openai": f"OpenAI ({self.OPENAI_MODEL})",
            "claude": f"Anthropic ({self.CLAUDE_MODEL})",
            "sim":    "Simulation (no API key)",
        }.get(self.mode, "Unknown")

    # ── Groq ──────────────────────────────────────────────────────────────────

    def _call_groq(self, system_prompt: str, user_message: str, max_tokens: int) -> str:
        import time
        max_tokens = min(max_tokens, 60)    # strict cap — 60 tokens/response keeps TPM under 6k
        for attempt in range(3):
            try:
                response = self._groq_client.chat.completions.create(
                    model=self.GROQ_MODEL,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=0.7,
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 4 ** attempt   # 1s, then 4s
                    print(f"[LLMBackend] Rate limit hit — waiting {wait}s (attempt {attempt+1}/3)")
                    time.sleep(wait)
                else:
                    print(f"[LLMBackend] Groq call failed: {e}")
                    return self._simulate(system_prompt, user_message)
        return self._simulate(system_prompt, user_message)

    # ── OpenAI ────────────────────────────────────────────────────────────────

    def _call_openai(self, system_prompt: str, user_message: str, max_tokens: int) -> str:
        try:
            response = self._openai_client.chat.completions.create(
                model=self.OPENAI_MODEL,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLMBackend] OpenAI call failed: {e}")
            return self._simulate(system_prompt, user_message)

    # ── Anthropic ─────────────────────────────────────────────────────────────

    def _call_claude(self, system_prompt: str, user_message: str, max_tokens: int) -> str:
        try:
            response = self._claude_client.messages.create(
                model=self.CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except Exception as e:
            print(f"[LLMBackend] Claude call failed: {e}")
            return self._simulate(system_prompt, user_message)

    # ── Simulation fallback ───────────────────────────────────────────────────

    _SIM_EXPLAIN = [
        "Let's break this down step by step. {topic} builds on what you already know. The key idea is to work from definitions toward applications — what do you know about it so far?",
        "Great question! For {topic}, let's start with the intuition before the formulas. Imagine explaining this to a friend in one sentence — what would you say?",
        "To understand {topic}, let's begin with a concrete example. Once you see it in action, the abstract definition makes much more sense.",
    ]
    _SIM_QUESTION_EASY = [
        "Quick check: can you state the basic definition of {topic} in your own words?",
        "What is the simplest case of {topic}? Start there before we go further.",
        "Before we continue — what's the first step when you encounter a {topic} problem?",
    ]
    _SIM_QUESTION_MEDIUM = [
        "Here's one at your level: given a standard {topic} problem, walk me through your approach step by step.",
        "Apply what you know: how does {topic} connect to what we covered earlier? Give me a concrete example.",
        "Try this: what happens when you apply the main technique of {topic} to a slightly unfamiliar situation?",
    ]
    _SIM_QUESTION_HARD = [
        "Challenge: can you derive or prove the key result for {topic} from first principles? Take your time.",
        "Advanced application — how does {topic} behave at the edge cases? Think carefully and show your reasoning.",
        "Deep dive: what are the fundamental limitations of the standard approach to {topic}, and when does it break down?",
    ]
    _SIM_HINT = [
        "Here's a nudge for {topic}: think about what the definition is actually saying, and apply it literally to this case.",
        "Hint: try working backwards from what the answer should look like. What constraints must it satisfy for {topic}?",
        "Consider this: which single property of {topic} is most relevant right now? Start there and the rest follows.",
    ]
    _SIM_EXAMPLE = [
        "Concrete example for {topic}: Step 1 — identify what's given. Step 2 — choose the right technique. Step 3 — apply it carefully. Step 4 — verify the answer makes sense.",
        "Real-world case for {topic}: imagine a practical scenario where this comes up. Here's how you'd solve it systematically...",
        "Worked example: let's take a typical {topic} problem and break it apart piece by piece so the pattern becomes clear.",
    ]
    _SIM_ENCOURAGE = [
        "You're making real progress with {topic}! It's genuinely hard — the fact that you're wrestling with it means you're learning.",
        "Don't worry, everyone finds {topic} tricky at first. Let's try a different angle and see if that clicks better.",
        "Keep going! {topic} will suddenly make sense once the core pattern clicks. You're much closer than you think.",
    ]

    def _simulate(self, system_prompt: str, user_message: str) -> str:
        topic = self._extract_topic(user_message)
        msg   = user_message.lower()
        if   "easy question"  in msg or "basic question" in msg: pool = self._SIM_QUESTION_EASY
        elif "hard question"  in msg or "challenge"      in msg: pool = self._SIM_QUESTION_HARD
        elif "question"       in msg or "quiz"           in msg: pool = self._SIM_QUESTION_MEDIUM
        elif "hint"           in msg or "stuck"          in msg: pool = self._SIM_HINT
        elif "example"        in msg or "show"           in msg: pool = self._SIM_EXAMPLE
        elif "encourage"      in msg or "motivat"        in msg: pool = self._SIM_ENCOURAGE
        else:                                                     pool = self._SIM_EXPLAIN
        return random.choice(pool).format(topic=topic)

    @staticmethod
    def _extract_topic(message: str) -> str:
        for kw in ["calculus", "algebra", "geometry", "statistics", "linear algebra",
                   "probability", "machine learning", "deep learning", "arithmetic",
                   "sorting", "graph", "dynamic programming", "physics", "kinematics"]:
            if kw in message.lower():
                return kw
        return "this topic"
