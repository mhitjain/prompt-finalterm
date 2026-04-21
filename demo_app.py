"""
Streamlit Interactive Demo — Dewey RL Adaptive Tutoring System

Graders/stakeholders can:
  1. Pick a student profile
  2. Watch the RL agent teach in real-time
  3. Compare trained vs untrained agent side-by-side
  4. See knowledge grow topic by topic
  5. Inspect WHAT the agent decided and WHY

Run:
    streamlit run demo_app.py
"""

import sys, os
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

from src.environment.tutorial_env import TutorialEnv, ACTION_NAMES
from src.environment.student_simulator import StudentProfile, TOPIC_NAMES
from src.agents.tutorial_agent import TutorialAgent
from src.agents.assessment_agent import AssessmentAgent
from src.agents.content_agent import ContentAgent
from src.agents.orchestrator import OrchestratorAgent
from src.rl.ppo import PPOConfig
from dewey.llm_backend import LLMBackend
from dewey.dewey_orchestrator import DeweyOrchestrator

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dewey RL Tutoring System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ───────────────────────────────────────────────────────────
ACTION_COLORS = {
    "EXPLAIN_CONCEPT":    "#4C72B0",
    "SHOW_EXAMPLE":       "#55A868",
    "ASK_EASY":           "#8ED06B",
    "ASK_MEDIUM":         "#F0A500",
    "ASK_HARD":           "#C44E52",
    "GIVE_HINT":          "#8172B2",
    "INCREASE_DIFFICULTY":"#DD8452",
    "DECREASE_DIFFICULTY":"#64B5CD",
    "SWITCH_TOPIC":       "#937860",
    "REVIEW_PREVIOUS":    "#DA8BC3",
    "ENCOURAGE":          "#8C8C8C",
}

PROFILE_EMOJI = {
    "FAST_LEARNER":     "⚡",
    "SLOW_LEARNER":     "🐢",
    "VISUAL_LEARNER":   "👁️",
    "PRACTICE_LEARNER": "💪",
}

# ── Session state helpers ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "step": 0,
        "done": False,
        "obs": None,
        "history": [],        # list of {action, topic, knowledge, engagement, reward, content}
        "system": None,
        "dewey": None,
        "trained": False,
        "total_reward": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _build_system(profile: StudentProfile, seed: int = 42) -> tuple:
    env = TutorialEnv(n_topics=10, max_steps=50, profile=profile, seed=seed)
    cfg = PPOConfig(lr=3e-4, hidden_dim=256, update_interval=1024)
    tutorial   = TutorialAgent(ppo_config=cfg)
    assessment = AssessmentAgent(alpha=1.0)
    content    = ContentAgent(alpha=1.0)
    orch = OrchestratorAgent(env, tutorial, assessment, content, verbose=False)
    backend = LLMBackend(
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
        api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
    )
    dewey = DeweyOrchestrator(env=env, rl_system=orch, backend=backend,
                               subject=st.session_state.get("subject","calculus"), verbose=False)
    return env, orch, dewey

# ── Knowledge bar chart ──────────────────────────────────────────────────────
def _knowledge_chart(knowledge: np.ndarray, title: str = "Student Knowledge"):
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#2ecc71" if k >= 0.85 else "#f39c12" if k >= 0.5 else "#e74c3c" for k in knowledge]
    bars = ax.barh(TOPIC_NAMES, knowledge, color=colors, edgecolor="white", height=0.6)
    ax.set_xlim(0, 1)
    ax.axvline(0.85, color="green", linestyle="--", alpha=0.4, label="Mastery (0.85)")
    ax.set_xlabel("Knowledge Level")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    # Annotate values
    for bar, k in zip(bars, knowledge):
        ax.text(min(k + 0.02, 0.95), bar.get_y() + bar.get_height()/2,
                f"{k:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    return fig

# ── Engagement gauge ─────────────────────────────────────────────────────────
def _engagement_gauge(eng: float):
    fig, ax = plt.subplots(figsize=(3, 2))
    color = "#2ecc71" if eng > 0.6 else "#f39c12" if eng > 0.3 else "#e74c3c"
    ax.barh(["Engagement"], [eng], color=color, height=0.4)
    ax.barh(["Engagement"], [1 - eng], left=[eng], color="#ecf0f1", height=0.4)
    ax.set_xlim(0, 1)
    ax.set_title(f"Engagement: {eng:.0%}", fontweight="bold", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    return fig

# ── Action history timeline ──────────────────────────────────────────────────
def _action_timeline(history: list):
    if not history:
        return None
    fig, ax = plt.subplots(figsize=(10, 2))
    for i, h in enumerate(history):
        color = ACTION_COLORS.get(h["action"], "#999")
        ax.barh(0, 1, left=i, color=color, height=0.6, edgecolor="white")
        if len(history) <= 20:
            ax.text(i + 0.5, 0, h["action"].replace("_", "\n"),
                    ha="center", va="center", fontsize=5, color="white", fontweight="bold")
    ax.set_xlim(0, max(len(history), 1))
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    ax.set_title("Teaching Action Timeline", fontweight="bold", fontsize=10)
    # Legend
    patches = [mpatches.Patch(color=c, label=a.replace("_"," "))
               for a, c in list(ACTION_COLORS.items())[:6]]
    ax.legend(handles=patches, loc="lower right", fontsize=6, ncol=3)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
_init_state()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 Dewey RL Tutor")
    st.markdown("**Reinforcement Learning for Adaptive Tutoring**")
    st.divider()

    subject = st.selectbox("Subject", ["calculus", "physics", "algorithms"], key="subject")

    profile_name = st.selectbox(
        "Student Profile",
        ["FAST_LEARNER", "SLOW_LEARNER", "VISUAL_LEARNER", "PRACTICE_LEARNER"],
        format_func=lambda x: f"{PROFILE_EMOJI[x]} {x.replace('_', ' ').title()}"
    )
    profile = StudentProfile[profile_name]

    st.divider()
    train_steps = st.slider("Pre-train RL agent (steps)", 0, 100000, 30000, 10000)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 New Session", use_container_width=True):
            env, orch, dewey = _build_system(profile)
            st.session_state.system  = (env, orch)
            st.session_state.dewey   = dewey
            st.session_state.trained = False
            st.session_state.step    = 0
            st.session_state.done    = False
            st.session_state.history = []
            st.session_state.total_reward = 0.0
            st.session_state.obs     = orch.reset()
            if train_steps > 0:
                with st.spinner(f"Training RL agent on {train_steps:,} student simulations..."):
                    dewey.train(total_timesteps=train_steps)
                st.session_state.trained = True
                st.success(f"✅ Trained on {train_steps:,} steps")
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            for k in ["step","done","obs","history","system","dewey","trained","total_reward"]:
                del st.session_state[k]
            st.rerun()

    st.divider()
    if st.session_state.trained:
        st.success("🧠 RL Policy: Trained")
    else:
        st.warning("🎲 RL Policy: Untrained (random)")

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("- PPO decides *what* to teach")
    st.markdown("- Thompson Sampling picks *difficulty*")
    st.markdown("- Ada/Newton/Grace deliver *content*")

# ── Main content ─────────────────────────────────────────────────────────────
st.title("🎓 Dewey RL Adaptive Tutoring System")
st.caption("Proximal Policy Optimisation + Linear Thompson Sampling | Dewey Framework (Humanitarians.AI)")

if st.session_state.system is None:
    st.info("👈 Click **New Session** in the sidebar to start a tutoring session.")
    st.markdown("""
    ### What you'll see:
    - **RL agent** selects teaching actions at every step (explain, ask question, hint, switch topic...)
    - **Student knowledge** grows topic by topic in real-time
    - **Engagement meter** shows student motivation
    - **Compare** trained vs untrained agent — see the difference RL makes
    """)

    # Show architecture diagram as text
    with st.expander("📐 System Architecture"):
        st.code("""
RL Layer (PPO + Thompson Sampling)
        ↓  decides WHAT action to take
Dewey Agents (Ada / Newton / Grace)
        ↓  generates actual educational content
Student Simulator (IRT-based)
        ↓  responds realistically, updates knowledge
Reward Function
        ↓  scores the teaching quality
PPO Policy Update  ←  learns from thousands of sessions
        """)
    st.stop()

env, orch = st.session_state.system
dewey     = st.session_state.dewey

# ── Step button ───────────────────────────────────────────────────────────────
col_step, col_auto, col_info = st.columns([2, 2, 4])

with col_step:
    step_clicked = st.button(
        "▶ Next Step",
        disabled=st.session_state.done,
        use_container_width=True,
        type="primary",
    )
with col_auto:
    run_all = st.button(
        "⏩ Run Full Session",
        disabled=st.session_state.done,
        use_container_width=True,
    )

if step_clicked or run_all:
    steps_to_run = 50 if run_all else 1
    for _ in range(steps_to_run):
        if st.session_state.done:
            break
        obs = st.session_state.obs
        action, topic_override = orch.act(obs, {})
        content = dewey._dispatch_to_dewey(action, TOPIC_NAMES[env.current_topic], topic_override)
        next_obs, reward, done, info = env.step(action, topic_override)
        orch.tutorial_agent.store_transition(reward, done)

        st.session_state.history.append({
            "step":       st.session_state.step + 1,
            "action":     ACTION_NAMES[action],
            "topic":      TOPIC_NAMES[env.current_topic],
            "knowledge":  env.student.true_knowledge.copy(),
            "engagement": info["engagement"],
            "reward":     reward,
            "content":    content or "",
            "mean_know":  info["mean_knowledge"],
            "n_mastered": info["n_mastered"],
        })
        st.session_state.total_reward += reward
        st.session_state.obs   = next_obs
        st.session_state.step += 1
        st.session_state.done  = done
    st.rerun()

# ── Live metrics row ──────────────────────────────────────────────────────────
if st.session_state.history:
    latest = st.session_state.history[-1]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Step",             f"{st.session_state.step} / 50")
    m2.metric("Mean Knowledge",   f"{latest['mean_know']:.3f}")
    m3.metric("Topics Mastered",  f"{latest['n_mastered']} / 10")
    m4.metric("Engagement",       f"{latest['engagement']:.0%}")
    m5.metric("Total Reward",     f"{st.session_state.total_reward:.1f}")

    if st.session_state.done:
        if env.student.is_disengaged():
            st.error("❌ Session ended — student disengaged.")
        else:
            st.success(f"✅ Session complete! Knowledge gained: {latest['mean_know']:.3f} | Mastered: {latest['n_mastered']}/10")

# ── Main panels ───────────────────────────────────────────────────────────────
if st.session_state.history:
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Knowledge Progress", "🤖 RL Decisions", "💬 Tutoring Transcript", "📈 Learning Curve"]
    )

    with tab1:
        latest = st.session_state.history[-1]
        st.pyplot(_knowledge_chart(latest["knowledge"], f"Student Knowledge — Step {st.session_state.step}"))
        col_e1, col_e2 = st.columns([1, 3])
        with col_e1:
            st.pyplot(_engagement_gauge(latest["engagement"]))
        with col_e2:
            # Knowledge gain over time
            if len(st.session_state.history) > 1:
                mean_knows = [h["mean_know"] for h in st.session_state.history]
                fig2, ax2 = plt.subplots(figsize=(6, 2.5))
                ax2.plot(range(1, len(mean_knows)+1), mean_knows, color="#2ecc71", linewidth=2)
                ax2.fill_between(range(1, len(mean_knows)+1), mean_knows, alpha=0.2, color="#2ecc71")
                ax2.set_xlabel("Step"); ax2.set_ylabel("Mean Knowledge")
                ax2.set_title("Knowledge Growth This Session", fontweight="bold")
                ax2.set_ylim(0, 1)
                plt.tight_layout()
                st.pyplot(fig2)

    with tab2:
        # Action timeline
        st.pyplot(_action_timeline(st.session_state.history))
        st.divider()

        # Last action detail
        latest = st.session_state.history[-1]
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Last RL Decision")
            color = ACTION_COLORS.get(latest["action"], "#999")
            st.markdown(
                f'<div style="background:{color};padding:12px;border-radius:8px;color:white;'
                f'font-size:18px;font-weight:bold;text-align:center">{latest["action"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"**Topic:** {latest['topic']}")
            st.markdown(f"**Reward received:** `{latest['reward']:+.2f}`")

        with col_b:
            # Action distribution pie
            from collections import Counter
            counts = Counter(h["action"] for h in st.session_state.history)
            fig3, ax3 = plt.subplots(figsize=(4, 3))
            labels = list(counts.keys())
            sizes  = list(counts.values())
            colors = [ACTION_COLORS.get(l, "#999") for l in labels]
            ax3.pie(sizes, labels=[l.replace("_","\n") for l in labels],
                    colors=colors, autopct="%1.0f%%", textprops={"fontsize": 7})
            ax3.set_title("Action Distribution", fontweight="bold", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig3)

    with tab3:
        st.markdown("### Tutoring Transcript")
        for h in reversed(st.session_state.history[-10:]):
            color = ACTION_COLORS.get(h["action"], "#999")
            with st.container():
                st.markdown(
                    f'<span style="background:{color};color:white;padding:2px 8px;'
                    f'border-radius:4px;font-size:12px">{h["action"]}</span> '
                    f'<span style="color:grey;font-size:12px">Step {h["step"]} | {h["topic"]}</span>',
                    unsafe_allow_html=True
                )
                if h["content"]:
                    st.markdown(f"> {h['content'][:300]}")
                st.caption(f"Knowledge: {h['mean_know']:.3f} | Engagement: {h['engagement']:.0%} | Reward: {h['reward']:+.2f}")
                st.divider()

    with tab4:
        rewards = [h["reward"] for h in st.session_state.history]
        cum_rewards = np.cumsum(rewards)
        mastered  = [h["n_mastered"] for h in st.session_state.history]

        fig4, (ax_r, ax_m) = plt.subplots(1, 2, figsize=(10, 3))
        ax_r.plot(cum_rewards, color="#3498db", linewidth=2)
        ax_r.fill_between(range(len(cum_rewards)), cum_rewards, alpha=0.15, color="#3498db")
        ax_r.set_title("Cumulative Reward", fontweight="bold")
        ax_r.set_xlabel("Step")

        ax_m.step(range(len(mastered)), mastered, color="#2ecc71", linewidth=2, where="post")
        ax_m.set_title("Topics Mastered Over Time", fontweight="bold")
        ax_m.set_xlabel("Step")
        ax_m.set_ylabel("Topics Mastered")
        ax_m.set_ylim(0, 10)

        plt.tight_layout()
        st.pyplot(fig4)

        # RL status banner
        st.divider()
        if st.session_state.trained:
            st.success(f"🧠 **Trained RL policy** — agent learned from {train_steps:,} simulated student sessions before this demo")
        else:
            st.warning("⚠️ **Untrained policy** — agent is picking actions randomly. Train it to see the difference!")
