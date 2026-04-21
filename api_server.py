"""
FastAPI Production Server — Dewey RL Adaptive Tutoring API

Exposes the trained RL tutoring system as a RESTful API.
Demonstrates production deployment readiness.

Run:
    uvicorn api_server:app --reload --port 8000

Endpoints:
    POST /session/new          — start new tutoring session
    POST /session/{id}/step    — take one tutoring step
    GET  /session/{id}/status  — get current session status
    POST /session/{id}/train   — trigger RL training (async)
    GET  /health               — health check
    GET  /metrics              — system metrics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List
import uuid
import os
import sys
import numpy as np
import time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from src.environment.tutorial_env import TutorialEnv
from src.environment.student_simulator import StudentProfile
from src.agents.tutorial_agent import TutorialAgent
from src.agents.assessment_agent import AssessmentAgent
from src.agents.content_agent import ContentAgent
from src.agents.orchestrator import OrchestratorAgent
from src.rl.ppo import PPOConfig
from dewey.llm_backend import LLMBackend
from dewey.dewey_orchestrator import DeweyOrchestrator


# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Dewey RL Tutoring API",
    description="Reinforcement Learning-enhanced adaptive tutoring system",
    version="1.0.0",
)

# In-memory session store (production would use Redis/DB)
_sessions: dict = {}
_training_status: dict = {}
_server_start = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class NewSessionRequest(BaseModel):
    subject: str = Field(default="calculus", description="Subject: calculus | physics | algorithms")
    profile_id: int = Field(default=0, ge=0, le=3, description="Student profile 0-3 (fast/slow/visual/practice)")
    max_steps: int = Field(default=50, ge=10, le=500, description="Max steps per session (10-500)")
    seed: int = Field(default=42)


class StepRequest(BaseModel):
    pass  # step uses GET-style, no body needed


class SessionStatus(BaseModel):
    session_id: str
    subject: str
    step: int
    total_reward: float
    mean_knowledge: float
    n_mastered: int
    engagement: float
    disengaged: bool
    current_topic: int
    knowledge_by_topic: List[float]
    last_action: Optional[str]
    last_content: Optional[str]


class StepResponse(BaseModel):
    session_id: str
    step: int
    action: str
    action_id: int
    reward: float
    done: bool
    content: str
    knowledge_gain: float
    engagement: float
    mean_knowledge: float
    n_mastered: int


class TrainRequest(BaseModel):
    timesteps: int = Field(default=50000, ge=1000, le=500000)


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    active_sessions: int
    version: str


class MetricsResponse(BaseModel):
    total_sessions_created: int
    active_sessions: int
    total_steps_taken: int
    mean_reward_across_sessions: float
    mean_knowledge_across_sessions: float


# Global counter for metrics
_total_sessions = 0
_total_steps = 0


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a fresh RL system
# ─────────────────────────────────────────────────────────────────────────────

def _build_system(subject: str, profile_id: int, seed: int, max_steps: int = 50):
    profile = StudentProfile(profile_id)
    env = TutorialEnv(n_topics=10, max_steps=max_steps, profile=profile, seed=seed)
    cfg = PPOConfig(lr=3e-4, hidden_dim=256, update_interval=1024)
    tutorial    = TutorialAgent(ppo_config=cfg)
    assessment  = AssessmentAgent(alpha=1.0)
    content     = ContentAgent(alpha=1.0)
    rl_system   = OrchestratorAgent(env, tutorial, assessment, content, verbose=False)

    backend = LLMBackend()  # auto-reads GROQ_API_KEY from .env

    dewey = DeweyOrchestrator(
        env=env,
        rl_system=rl_system,
        backend=backend,
        subject=subject,
        verbose=False,
    )
    return env, rl_system, dewey


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _server_start, 1),
        active_sessions=len(_sessions),
        version="1.0.0",
    )


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    rewards = [s["total_reward"] for s in _sessions.values()]
    knowledges = [s["mean_knowledge"] for s in _sessions.values()]
    return MetricsResponse(
        total_sessions_created=_total_sessions,
        active_sessions=len(_sessions),
        total_steps_taken=_total_steps,
        mean_reward_across_sessions=float(np.mean(rewards)) if rewards else 0.0,
        mean_knowledge_across_sessions=float(np.mean(knowledges)) if knowledges else 0.0,
    )


@app.post("/session/new")
def new_session(req: NewSessionRequest):
    global _total_sessions
    session_id = str(uuid.uuid4())[:8]
    env, rl_system, dewey = _build_system(req.subject, req.profile_id, req.seed, req.max_steps)
    dewey.reset()
    _sessions[session_id] = {
        "env": env,
        "rl_system": rl_system,
        "dewey": dewey,
        "subject": req.subject,
        "profile_id": req.profile_id,
        "step": 0,
        "total_reward": 0.0,
        "mean_knowledge": 0.0,
        "n_mastered": 0,
        "last_action": None,
        "last_content": None,
        "done": False,
        "created_at": datetime.utcnow().isoformat(),
    }
    _total_sessions += 1
    return {"session_id": session_id, "message": f"Session started for subject={req.subject}"}


@app.post("/session/{session_id}/step", response_model=StepResponse)
def step_session(session_id: str, background_tasks: BackgroundTasks):
    global _total_steps
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    s = _sessions[session_id]
    if s["done"]:
        raise HTTPException(status_code=400, detail="Session is already complete")

    env = s["env"]
    dewey = s["dewey"]

    prev_knowledge = float(np.mean(env._get_obs()[:10]))

    # Take one step using the Dewey orchestrator
    result = dewey.step()

    obs = env._get_obs()
    mean_knowledge = float(np.mean(obs[:10]))
    knowledge_gain = mean_knowledge - prev_knowledge
    engagement = float(obs[21])

    ACTION_NAMES = [
        "EXPLAIN_CONCEPT", "SHOW_EXAMPLE", "ASK_EASY",
        "ASK_MEDIUM", "ASK_HARD", "GIVE_HINT",
        "INCREASE_DIFFICULTY", "DECREASE_DIFFICULTY",
        "SWITCH_TOPIC", "REVIEW_PREVIOUS", "ENCOURAGE",
    ]
    action_id = result.get("action", 0)
    action_name = ACTION_NAMES[action_id] if action_id < len(ACTION_NAMES) else "UNKNOWN"

    s["step"] += 1
    s["total_reward"] += result.get("reward", 0.0)
    s["mean_knowledge"] = mean_knowledge
    s["n_mastered"] = int(sum(obs[:10] > 0.8))
    s["last_action"] = action_name
    s["last_content"] = result.get("content", "")
    s["done"] = result.get("done", False)

    _total_steps += 1

    return StepResponse(
        session_id=session_id,
        step=s["step"],
        action=action_name,
        action_id=action_id,
        reward=float(result.get("reward", 0.0)),
        done=s["done"],
        content=result.get("content", ""),
        knowledge_gain=round(knowledge_gain, 4),
        engagement=round(engagement, 3),
        mean_knowledge=round(mean_knowledge, 3),
        n_mastered=s["n_mastered"],
    )


@app.get("/session/{session_id}/status", response_model=SessionStatus)
def session_status(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = _sessions[session_id]
    env = s["env"]
    obs = env._get_obs()
    return SessionStatus(
        session_id=session_id,
        subject=s["subject"],
        step=s["step"],
        total_reward=round(s["total_reward"], 2),
        mean_knowledge=round(s["mean_knowledge"], 3),
        n_mastered=s["n_mastered"],
        engagement=round(float(obs[21]), 3),
        disengaged=bool(obs[22] > 0.5),
        current_topic=int(env.current_topic),
        knowledge_by_topic=[round(float(obs[i]), 3) for i in range(10)],
        last_action=s["last_action"],
        last_content=s.get("last_content"),
    )


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
    return {"message": f"Session {session_id} deleted"}


@app.post("/session/{session_id}/train")
def train_session(session_id: str, req: TrainRequest, background_tasks: BackgroundTasks):
    """Trigger RL training for this session's environment (async background task)."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    _training_status[session_id] = {"status": "running", "started_at": datetime.utcnow().isoformat()}

    def _do_train():
        s = _sessions.get(session_id)
        if not s:
            return
        try:
            result = s["dewey"].train(total_timesteps=req.timesteps, verbose=False)
            _training_status[session_id] = {
                "status": "complete",
                "mean_reward": result.get("mean_reward_last20", 0),
                "finished_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            _training_status[session_id] = {"status": "error", "message": str(e)}

    background_tasks.add_task(_do_train)
    return {"message": f"Training started for {req.timesteps} timesteps", "session_id": session_id}


@app.get("/session/{session_id}/training-status")
def training_status(session_id: str):
    status = _training_status.get(session_id, {"status": "not_started"})
    return {"session_id": session_id, **status}


# ─────────────────────────────────────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
