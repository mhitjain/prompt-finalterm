# Dewey RL: Reinforcement Learning-Enhanced Adaptive Tutoring System

**Course:** Reinforcement Learning for Agentic AI Systems  
**Framework:** [Dewey (Humanitarians.AI)](https://github.com/Humanitariansai/Dewey) — RL Integration

---

## Overview

This project extends the **Dewey Framework** (Humanitarians.AI) with reinforcement learning to create a fully adaptive, personalised tutoring system. Dewey defines tutorial agents (Ada for Calculus, Newton for Physics, Grace for Algorithms) as specification documents. We implement these agents as executable Python code and add a two-algorithm RL layer that learns to optimise pedagogical strategy through experience.

### The Key Idea

```
RL Layer  (PPO + Thompson Sampling)  →  decides the PEDAGOGICAL STRATEGY
    ↓                                    (explain? ask? hint? switch topic? encourage?)
Dewey Agents (Ada / Newton / Grace)  →  generates the actual CONTENT
    ↓                                    (powered by Groq/OpenAI/Claude or simulation)
Student receives personalised, adaptive lesson
```

**Why this matters:** A Fixed Script (EXPLAIN→ASK→HINT loop) achieves 0.16 engagement after 150 steps — students give up. The trained RL maintains 0.64 engagement and accumulates **34% more knowledge gain** and **746% more cumulative reward** over an extended session.

### Two RL Algorithms Implemented

| Approach | Algorithm | Agent | Purpose |
|---|---|---|---|
| **Policy Gradient** | PPO (Schulman et al., 2017) | `TutorialAgent` | High-level teaching strategy — 11 discrete actions |
| **Exploration** | Linear Thompson Sampling | `AssessmentAgent`, `ContentAgent` | Adaptive question difficulty and content style |

---

## Key Results (150-step sessions, averaged over 20 students)

| System | Engagement | Knowledge Gain | Cumulative Reward |
|---|---|---|---|
| Fixed Script | 0.16 ❌ | 0.77 | 12.9 |
| Untrained RL | 0.63 | 3.19 | 56.4 |
| **Trained RL (PPO+TS)** | **0.64** | **4.26 (+34%)** | **108.8 (+746% vs Fixed)** |

> At 50 steps, all systems look similar. At 150 steps, Fixed Script engagement collapses while the trained RL sustains learning — this is the core RL advantage.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Configure LLM backend (optional)

Copy the example env file and add your API key:

```bash
cp .env.example .env
# Edit .env and add one of:
# GROQ_API_KEY=gsk_...        ← recommended (free tier, fast)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# Leave blank for simulation mode (no API key needed)
```

Auto-detection priority: **Groq → OpenAI → Anthropic → Simulation**

---

## Running the Demos

### Main Demo: Live RL vs Baseline Comparison
```bash
streamlit run compare_app.py
# Visit http://localhost:8501
```
Shows three columns (Untrained RL | Fixed Script | Trained RL) teaching the same student side-by-side with real LLM content. Includes the 150-step long-horizon chart at the bottom.

### Production API Demo
```bash
# Terminal 1 — start the API server
uvicorn api_server:app --reload --port 8000

# Terminal 2 — start the frontend UI
streamlit run api_ui.py --server.port 8503
# Visit http://localhost:8503
```
Full REST API with session management. Shows what a university LMS integration looks like.

### Single-Agent Dewey Demo
```bash
# Simulation mode (no API key needed)
python dewey_demo.py --subject calculus --steps 12

# With training first
python dewey_demo.py --subject calculus --train --timesteps 80000 --steps 12

# Physics or Algorithms
python dewey_demo.py --subject physics --steps 10
python dewey_demo.py --subject algorithms --steps 10
```

### API Explorer (Swagger UI)
```bash
uvicorn api_server:app --reload --port 8000
# Visit http://localhost:8000/docs
```

---

## Running Experiments

```bash
# Full experiment: train + evaluate + generate all figures
python experiments/run_all.py --timesteps 150000 --seeds 2

# 150-step learning curves comparison (the key chart)
python experiments/learning_curves_comparison.py

# Policy analysis: action heatmap, curriculum discovery, significance tests
python experiments/policy_analysis.py --timesteps 40000 --eval_eps 20

# Educational baseline comparison (vs Mastery Learning, ZPD, Spaced Repetition)
python experiments/baselines.py

# Ablation study: full system vs PPO-only vs random
python experiments/ablation.py --timesteps 80000

# Run tests
pytest tests/ -v
```

All outputs save to `results/`:
- `figures/` — learning curves, comparison bars, heatmaps, policy analysis
- `ppo_checkpoints/` — trained model weights
- `summary.json` — aggregated metrics

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      OrchestratorAgent                           │
│                                                                  │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ TutorialAgent│  │ AssessmentAgent   │  │ ContentAgent     │  │
│  │   (PPO)      │  │ (Thompson Samp.) │  │ (Thompson Samp.) │  │
│  │              │  │                  │  │                  │  │
│  │ 11 actions   │  │ 3 arms (easy/    │  │ 5 arms (explain/ │  │
│  │ - explain    │  │   medium/hard)   │  │   example/visual/│  │
│  │ - ask Q      │  │                  │  │   review/hint)   │  │
│  │ - hint, etc. │  │                  │  │                  │  │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬─────────┘  │
│         └──────────────────►│◄───────────────────-┘            │
│                             ▼                                   │
│                     ┌───────────────┐                           │
│                     │  TutorialEnv  │                           │
│                     │(IRT+BKT Sim.) │                           │
│                     └───────────────┘                           │
│                                                                  │
│  Custom Tools: KnowledgeGraph | DifficultyEstimator | PerfTracker│
└──────────────────────────────────────────────────────────────────┘
         ↓ RL decisions
┌──────────────────────────────────────────────────────────────────┐
│                    DeweyOrchestrator                             │
│  RL action → Ada (Calculus) / Newton (Physics) / Grace (Algos)  │
│  LLMBackend: Groq (llama-3.1-8b) / OpenAI / Anthropic / Sim.   │
└──────────────────────────────────────────────────────────────────┘
         ↓ REST API
┌──────────────────────────────────────────────────────────────────┐
│              FastAPI Production Server (api_server.py)          │
│  POST /session/new  ·  POST /session/{id}/step                  │
│  GET  /session/{id}/status  ·  GET /health  ·  GET /metrics     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Reinforcement Learning Design

### PPO (Policy Gradient)

**State Space** (26-dimensional):
```
[0:10]  noisy per-topic knowledge estimates
[10:20] one-hot current topic
[20]    difficulty level
[21]    engagement score
[22]    session progress
[23]    consecutive correct / 10
[24]    consecutive wrong / 10
[25]    prerequisite readiness
```

**Action Space** (11 discrete actions):
```
0  EXPLAIN_CONCEPT        6  INCREASE_DIFFICULTY
1  SHOW_EXAMPLE           7  DECREASE_DIFFICULTY
2  ASK_EASY               8  SWITCH_TOPIC
3  ASK_MEDIUM             9  REVIEW_PREVIOUS
4  ASK_HARD              10  ENCOURAGE
5  GIVE_HINT
```

**Reward Function**:
```
r_t = -0.01  (time penalty — encourages efficiency)
    + 5.0 × knowledge_gain        (dense: per-step learning)
    + 0.5 × correct               (correct answer bonus)
    - 0.3 × wrong                 (wrong answer penalty)
    + 0.3 × engagement            (engagement maintenance)
    - 5.0 × [engagement < 0.15]   (disengagement penalty)
    + 10.0 × [topic mastered]     (sparse mastery bonus at ≥0.85)
    + 1.0 × prereq × k_gain       (curriculum shaping)
    + 2.0 × sum(knowledge) [terminal bonus]
```

**PPO Update** (Schulman et al., 2017):
```
L^CLIP(θ) = E_t [ min(r_t(θ) Â_t,  clip(r_t(θ), 1-ε, 1+ε) Â_t) ]
```
Advantage estimates use GAE-λ (γ=0.99, λ=0.95, ε=0.2, lr=3×10⁻⁴)

### Linear Thompson Sampling (Contextual Bandits)

Models reward as linear function of context: `E[r|x,a] = xᵀθ_a`

Posterior update: `B_a = αI + XₐᵀXₐ`, `μ_a = Bₐ⁻¹fₐ`, sample `θ̃_a ~ N(μ_a, α·Bₐ⁻¹)`

---

## Custom Tools

| Tool | File | Description |
|---|---|---|
| `KnowledgeGraphTool` | `src/tools/knowledge_graph.py` | NetworkX DAG; `best_next_topic()`, `transfer_potential()`, topological sort |
| `DifficultyEstimatorTool` | `src/tools/difficulty_estimator.py` | IRT 2PL; EAP ability estimation, ZPD-optimal difficulty, Fisher information |
| `PerformanceTrackerTool` | `src/tools/performance_tracker.py` | Step/episode analytics; smoothed learning curves with SEM, pandas export |

---

## Project Structure

```
Take-home-final/
├── dewey/                          ← Dewey Framework integration
│   ├── ada_agent.py                # Ada: Socratic calculus tutor
│   ├── newton_agent.py             # Newton: Physics tutor
│   ├── grace_agent.py              # Grace: Algorithms tutor
│   ├── dewey_orchestrator.py       # Connects RL decisions → Dewey agents
│   └── llm_backend.py              # Groq/OpenAI/Anthropic/Simulation backend
├── src/
│   ├── environment/
│   │   ├── tutorial_env.py         # Gym-style RL environment
│   │   ├── student_simulator.py    # IRT + BKT student model (4 profiles)
│   │   └── reward_function.py      # Shaped reward engineering
│   ├── rl/
│   │   ├── ppo.py                  # PPO with GAE-λ
│   │   ├── networks.py             # Actor-Critic (LayerNorm + Tanh)
│   │   ├── buffer.py               # On-policy rollout buffer
│   │   └── contextual_bandits.py   # Linear Thompson Sampling
│   ├── agents/
│   │   ├── orchestrator.py         # Multi-agent coordinator
│   │   ├── tutorial_agent.py       # PPO teaching agent
│   │   ├── assessment_agent.py     # Bandit question difficulty
│   │   └── content_agent.py        # Bandit content style
│   └── tools/
│       ├── knowledge_graph.py      # Custom KG tool
│       ├── difficulty_estimator.py # Custom IRT tool
│       └── performance_tracker.py  # Custom analytics tool
├── experiments/
│   ├── learning_curves_comparison.py  # Key 150-step chart (averaged 20 students)
│   ├── run_all.py                  # Full train+evaluate pipeline
│   ├── evaluate.py                 # Per-profile evaluation
│   ├── ablation.py                 # Ablation study
│   ├── baselines.py                # 5 educational baselines
│   └── policy_analysis.py          # Heatmaps, curriculum discovery, stats
├── compare_app.py                  ← PRIMARY DEMO: 3-way live comparison UI
├── api_server.py                   ← FastAPI production REST API
├── api_ui.py                       ← API frontend (shows LMS integration)
├── demo_app.py                     ← Single-agent interactive demo
├── dewey_demo.py                   ← CLI Dewey agent demo
├── results/
│   ├── figures/                    # All generated charts (PNG)
│   └── ppo_checkpoints/            # Trained model weights
├── tests/                          # 36 unit tests (pytest)
├── .env.example                    # API key template
├── config.yaml
├── requirements.txt
└── REPORT.md                       # Full technical report
```

---

## Ethical Considerations

- **Fairness**: Evaluated across 4 learner profiles to ensure equitable performance
- **Transparency**: All RL decisions logged and auditable via `/session/{id}/status`
- **No Real Student Data**: Simulator is fully synthetic — IRT + BKT models
- **Graceful Degradation**: Simulation fallback ensures safe operation without API keys
- **Privacy**: API keys stored in `.env` (gitignored), never exposed in UI
