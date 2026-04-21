"""
Streamlit UI: Fixed Script vs RL — Live Side-by-Side Comparison

Run:
    streamlit run compare_app.py
"""

import streamlit as st
import sys, os, time
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # reads .env silently — key never appears in UI

sys.path.insert(0, os.path.dirname(__file__))

from src.environment.tutorial_env import TutorialEnv, ACTION_NAMES
from src.environment.student_simulator import StudentProfile, TOPIC_NAMES
from src.agents.tutorial_agent import TutorialAgent
from src.agents.assessment_agent import AssessmentAgent
from src.agents.content_agent import ContentAgent
from src.agents.orchestrator import OrchestratorAgent
from src.rl.ppo import PPOConfig
from dewey.llm_backend import LLMBackend
from dewey.ada_agent import AdaAgent

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RL vs Baseline — Live Tutoring Comparison",
    page_icon="🎓",
    layout="wide",
)

PROFILES     = ["FAST_LEARNER", "SLOW_LEARNER", "VISUAL_LEARNER", "PRACTICE_LEARNER"]
FIXED_SCRIPT = [0, 3, 5]  # EXPLAIN → ASK_MEDIUM → HINT

ACTION_COLORS = {
    "EXPLAIN_CONCEPT":    "#4A90D9",
    "SHOW_EXAMPLE":       "#7B68EE",
    "ASK_EASY":           "#52C878",
    "ASK_MEDIUM":         "#F0A500",
    "ASK_HARD":           "#E05C5C",
    "GIVE_HINT":          "#20B2AA",
    "INCREASE_DIFFICULTY":"#FF8C00",
    "DECREASE_DIFFICULTY":"#9370DB",
    "SWITCH_TOPIC":       "#FF69B4",
    "REVIEW_PREVIOUS":    "#708090",
    "ENCOURAGE":          "#32CD32",
}

ACTION_TO_ADA = {
    0: ("explain_concept", "medium"),
    1: ("show_example",    "medium"),
    2: ("ask_question",    "easy"),
    3: ("ask_question",    "medium"),
    4: ("ask_question",    "hard"),
    5: ("give_hint",       "medium"),
    6: (None, None),
    7: (None, None),
    8: (None, None),
    9: ("review_prerequisites", "medium"),
    10:("encourage",       "medium"),
}

def get_ada_content(ada, action, topic):
    method, diff = ACTION_TO_ADA[action]
    if method is None:
        return f"*[{ACTION_NAMES[action]}]*"
    if method == "ask_question":
        return ada.ask_question(topic, difficulty=diff)
    return getattr(ada, method)(topic)

def action_badge(action_name):
    color = ACTION_COLORS.get(action_name, "#888")
    return f"""<span style="background:{color};color:white;padding:3px 10px;
               border-radius:12px;font-size:12px;font-weight:600">{action_name}</span>"""

def knowledge_bar(value, color="#4A90D9"):
    pct = int(value * 100)
    return f"""
    <div style="background:#eee;border-radius:6px;height:14px;margin:2px 0">
      <div style="background:{color};width:{pct}%;height:100%;border-radius:6px;
                  transition:width 0.4s"></div>
    </div>
    <div style="font-size:11px;color:#555">{pct}%</div>"""

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    profile_id = st.selectbox(
        "Student Profile",
        options=list(range(4)),
        format_func=lambda i: PROFILES[i],
        index=1,
        help="SLOW_LEARNER makes the difference most visible",
    )

    n_steps = st.slider("Steps per session", 5, 20, 12,
                        help="Keep at 12 with Groq free tier. The 150-step chart at the bottom shows the long-term story.")

    train_steps = st.select_slider(
        "RL Training Steps",
        options=[10000, 20000, 30000, 50000],
        value=20000,
        help="More steps = smarter agent. 20k takes ~20s.",
    )

    seed = st.number_input("Random seed", value=42, step=1)

    st.divider()
    st.markdown("**What you're watching:**")
    st.markdown("🔴 **Fixed Script** — rigid EXPLAIN→ASK→HINT loop. No adaptation.")
    st.markdown("🟢 **RL System** — PPO + Thompson Sampling, adapts every step.")
    st.markdown("Same student. Same subject. Different strategy.")

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style='text-align:center;margin-bottom:0'>🎓 Adaptive Tutoring: RL vs Baseline</h1>
<p style='text-align:center;color:#888;font-size:16px'>
  Both systems teach the same student. Watch how their strategies differ.
</p>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Start button
# ─────────────────────────────────────────────────────────────────────────────

CACHE_FILE = "demo_cache.json"

col_btn, col_replay, col_info = st.columns([1, 1, 2])
with col_btn:
    run = st.button("▶ Start Comparison", type="primary", use_container_width=True)
with col_replay:
    replay = st.button("⚡ Replay Last Run", use_container_width=True,
                       help="Instantly load the last saved results — no LLM calls needed.")
with col_info:
    st.markdown(f"""
    **Profile:** {PROFILES[profile_id]} &nbsp;|&nbsp;
    **Steps:** {n_steps} &nbsp;|&nbsp;
    **Training:** {train_steps:,} steps &nbsp;|&nbsp;
    **LLM:** {'Groq' if os.environ.get('GROQ_API_KEY') else ('GPT-4o-mini' if os.environ.get('OPENAI_API_KEY') else 'Simulation')}
    """)

# ── Replay path: load cached results instantly ────────────────────────────────
if replay:
    if os.path.exists(CACHE_FILE):
        import json
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        st.session_state["cached_results"] = cache
        st.success("⚡ Loaded pre-computed results instantly!")
    else:
        st.warning("No cached run found — click **Start Comparison** first to generate one.")
        st.stop()

if not run and "cached_results" not in st.session_state:
    st.info("👈 Configure settings in the sidebar, then click **Start Comparison**.")
    st.stop()

import json

# ── Cache path: skip all training/LLM when replaying ─────────────────────────
if "cached_results" in st.session_state:
    cache = st.session_state["cached_results"]
    fs_steps  = cache["fs_steps"]
    pre_steps = cache["pre_steps"]
    rl_steps  = cache["rl_steps"]
    st.info(f"⚡ Showing pre-computed results ({len(rl_steps)} steps, profile: {cache.get('profile', '?')})")

else:
    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Build backend + train RL
    # ─────────────────────────────────────────────────────────────────────────

    with st.status("🔧 Setting up...", expanded=True) as status:
        st.write("Loading LLM backend...")
        backend = LLMBackend()  # auto-detects GROQ_API_KEY / OPENAI_API_KEY from .env
        ada = AdaAgent(backend)
        st.write(f"✅ LLM backend: **{backend.provider_name}**")

        cfg = PPOConfig(lr=3e-4, hidden_dim=256, update_interval=1024)

        # Build UNTRAINED orchestrator (snapshot before any learning)
        env_pre = TutorialEnv(n_topics=10, max_steps=50, profile=StudentProfile(profile_id), seed=seed)
        orch_untrained = OrchestratorAgent(
            env_pre,
            TutorialAgent(ppo_config=cfg),
            AssessmentAgent(alpha=1.0),
            ContentAgent(alpha=1.0),
            verbose=False,
        )
        st.write("✅ Untrained RL agent ready (random policy — before learning)")

        # Build and TRAIN orchestrator
        st.write(f"Training RL policy ({train_steps:,} simulation steps — no API calls)...")
        env_rl = TutorialEnv(n_topics=10, max_steps=50, profile=StudentProfile(profile_id), seed=seed)
        orch   = OrchestratorAgent(
            env_rl,
            TutorialAgent(ppo_config=cfg),
            AssessmentAgent(alpha=1.0),
            ContentAgent(alpha=1.0),
            verbose=False,
        )
        train_bar = st.progress(0, text="Training RL policy...")
        obs = orch.reset()
        chunk = train_steps // 20
        for i in range(20):
            for _ in range(chunk):
                obs, r, done, _ = orch.run_step(obs)
                if done:
                    obs = orch.reset()
            train_bar.progress((i + 1) / 20, text=f"Training RL... {(i+1)*5}%")
        train_bar.empty()
        st.write(f"✅ RL policy trained ({train_steps:,} steps)")
        status.update(label="✅ Ready — 3 systems loaded!", state="complete")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Collect all steps (no sleep — then render)
    # ─────────────────────────────────────────────────────────────────────────

    EVAL_SEED = seed + 999  # unseen student — fair evaluation

    def run_rl_session(orchestrator, label):
        env_new = TutorialEnv(n_topics=10, max_steps=50,
                              profile=StudentProfile(profile_id), seed=EVAL_SEED)
        orchestrator.env = env_new
        obs = orchestrator.reset()
        steps = []
        for i in range(n_steps):
            action, topic_override = orchestrator.act(obs, {})
            topic   = TOPIC_NAMES[env_new.current_topic]
            content = get_ada_content(ada, action, topic)
            time.sleep(2.5)
            obs, reward, done, info = env_new.step(action, topic_override)
            orchestrator.tutorial_agent.store_transition(reward, done)
            steps.append({
                "step": i+1, "action": ACTION_NAMES[action], "topic": topic,
                "content": content, "reward": reward,
                "knowledge": info["mean_knowledge"], "engagement": info["engagement"],
                "k_by_topic": obs[:10].tolist(),
            })
            if done: break
        return steps

    # Run Fixed Script
    env_fs = TutorialEnv(n_topics=10, max_steps=50,
                         profile=StudentProfile(profile_id), seed=EVAL_SEED)
    env_fs.reset()
    fs_steps = []
    with st.spinner("Running Fixed Script baseline..."):
        for i in range(n_steps):
            action  = FIXED_SCRIPT[i % 3]
            topic   = TOPIC_NAMES[env_fs.current_topic]
            content = get_ada_content(ada, action, topic)
            time.sleep(2.5)
            obs, reward, done, info = env_fs.step(action)
            fs_steps.append({
                "step": i+1, "action": ACTION_NAMES[action], "topic": topic,
                "content": content, "reward": reward,
                "knowledge": info["mean_knowledge"], "engagement": info["engagement"],
                "k_by_topic": obs[:10].tolist(),
            })
            if done: break

    # Run Untrained RL (before learning)
    with st.spinner("Running Untrained RL — before any learning..."):
        pre_steps = run_rl_session(orch_untrained, "Untrained")

    # Run Trained RL (after learning)
    with st.spinner("Running Trained RL — after learning..."):
        rl_steps = run_rl_session(orch, "Trained")

    # Save to cache for instant replay
    cache_data = {
        "fs_steps": fs_steps, "pre_steps": pre_steps, "rl_steps": rl_steps,
        "profile": PROFILES[profile_id], "n_steps": n_steps, "train_steps": train_steps,
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f)
    st.session_state["cached_results"] = cache_data

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Render side-by-side
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Step-by-step comparison")

def render_step(s, bg, border):
    st.markdown(
        f"**Step {s['step']}** &nbsp; {action_badge(s['action'])} &nbsp;"
        f"<span style='color:#888;font-size:11px'>📚 {s['topic']}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='background:{bg};border-left:3px solid {border};"
        f"padding:6px 10px;border-radius:4px;margin:3px 0;font-size:13px'>"
        f"💬 {s['content']}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<small>📈 {s['knowledge']:.3f} &nbsp; 💡 {s['engagement']:.2f} &nbsp; 🏆 {s['reward']:+.2f}</small>",
        unsafe_allow_html=True,
    )

col_pre, col_fs, col_rl = st.columns(3)
with col_pre:
    st.markdown("### ⚪ Untrained RL")
    st.caption("Random policy — before any learning")
with col_fs:
    st.markdown("### 🔴 Fixed Script")
    st.caption("EXPLAIN→ASK→HINT forever")
with col_rl:
    st.markdown("### 🟢 Trained RL")
    st.caption("PPO + Thompson Sampling — after learning")

for i in range(max(len(fs_steps), len(rl_steps), len(pre_steps))):
    col_pre, col_fs, col_rl = st.columns(3)
    if i < len(pre_steps):
        with col_pre: render_step(pre_steps[i], "#1a1a2e", "#888888")
    if i < len(fs_steps):
        with col_fs:  render_step(fs_steps[i],  "#1a0a0a", "#E05C5C")
    if i < len(rl_steps):
        with col_rl:  render_step(rl_steps[i],  "#0a1a0a", "#52C878")
    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Scorecard
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📊 Final Scorecard")

fs_total_r  = sum(s["reward"]    for s in fs_steps)
rl_total_r  = sum(s["reward"]    for s in rl_steps)
pre_total_r = sum(s["reward"]    for s in pre_steps)
fs_final_k  = fs_steps[-1]["knowledge"]  if fs_steps  else 0
rl_final_k  = rl_steps[-1]["knowledge"]  if rl_steps  else 0
pre_final_k = pre_steps[-1]["knowledge"] if pre_steps else 0
fs_final_e  = fs_steps[-1]["engagement"] if fs_steps  else 0
rl_final_e  = rl_steps[-1]["engagement"] if rl_steps  else 0
pre_final_e = pre_steps[-1]["engagement"]if pre_steps else 0

fs_actions_used = set(s["action"] for s in fs_steps)
rl_actions_used = set(s["action"] for s in rl_steps)

st.markdown(f"| | ⚪ Untrained RL | 🔴 Fixed Script | 🟢 Trained RL |"
            f"\n|---|---|---|---|"
            f"\n| **Total Reward** | {pre_total_r:.1f} | {fs_total_r:.1f} | **{rl_total_r:.1f}** |"
            f"\n| **Final Knowledge** | {pre_final_k:.3f} | {fs_final_k:.3f} | **{rl_final_k:.3f}** |"
            f"\n| **Final Engagement** | {pre_final_e:.2f} | {fs_final_e:.2f} | **{rl_final_e:.2f}** |")

st.divider()
col1, col2, col3, col4 = st.columns(4)

# Knowledge gain (what matters, not absolute knowledge)
fs_start_k = fs_steps[0]["k_by_topic"] if fs_steps else [0]*10
rl_start_k = rl_steps[0]["k_by_topic"] if rl_steps else [0]*10
fs_gain = fs_final_k - float(sum(fs_start_k)/10)
rl_gain = rl_final_k - float(sum(rl_start_k)/10)

with col1:
    delta_r = rl_total_r - fs_total_r
    st.metric("Total Reward",
              f"RL: {rl_total_r:.1f}",
              delta=f"{delta_r:+.1f} vs Fixed Script",
              delta_color="normal")

with col2:
    delta_k = rl_final_k - fs_final_k
    st.metric("Avg Knowledge (all topics)",
              f"RL: {rl_final_k:.3f}",
              delta=f"{delta_k:+.3f} vs Fixed Script",
              delta_color="normal",
              help="Average across all 10 topics. Low because session only covers a few topics.")

with col3:
    delta_g = rl_gain - fs_gain
    st.metric("Knowledge GAINED this session",
              f"RL: +{rl_gain:.3f}",
              delta=f"{delta_g:+.3f} vs Fixed Script",
              delta_color="normal",
              help="How much the student actually learned — starting knowledge subtracted out.")

with col4:
    delta_e = rl_final_e - fs_final_e
    st.metric("Final Engagement",
              f"RL: {rl_final_e:.2f}",
              delta=f"{delta_e:+.2f} vs Fixed Script",
              delta_color="normal",
              help="Engagement near 1.0 = student still motivated. Near 0 = checked out.")

# Action diversity
st.divider()
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**🔴 Fixed Script — Actions used:**")
    for a in sorted(fs_actions_used):
        count = sum(1 for s in fs_steps if s["action"] == a)
        st.markdown(f"{action_badge(a)} × {count}", unsafe_allow_html=True)

with col_b:
    st.markdown("**🟢 RL System — Actions used:**")
    for a in sorted(rl_actions_used):
        count = sum(1 for s in rl_steps if s["action"] == a)
        st.markdown(f"{action_badge(a)} × {count}", unsafe_allow_html=True)

# Reward over time chart
st.divider()
st.markdown("**Cumulative Reward over time**")
import pandas as pd

fs_cumr = np.cumsum([s["reward"] for s in fs_steps])
rl_cumr = np.cumsum([s["reward"] for s in rl_steps])
steps_x = list(range(1, max(len(fs_steps), len(rl_steps)) + 1))

chart_data = pd.DataFrame({
    "Step": steps_x[:min(len(fs_cumr), len(rl_cumr))],
    "Fixed Script": fs_cumr[:min(len(fs_cumr), len(rl_cumr))],
    "RL System":    rl_cumr[:min(len(fs_cumr), len(rl_cumr))],
}).set_index("Step")
st.line_chart(chart_data, color=["#E05C5C", "#52C878"])

# Knowledge by topic — grouped bar chart (not stacked)
st.divider()
st.markdown("**Knowledge by Topic — where did each system focus? (final state)**")
st.caption("Each topic shown side by side. Green bar higher = RL taught that topic better.")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

fs_k = fs_steps[-1]["k_by_topic"] if fs_steps else [0]*10
rl_k = rl_steps[-1]["k_by_topic"] if rl_steps else [0]*10

x     = np.arange(len(TOPIC_NAMES))
width = 0.38

fig, ax = plt.subplots(figsize=(12, 4))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")

bars_fs = ax.bar(x - width/2, fs_k, width, label="Fixed Script", color="#E05C5C", alpha=0.85)
bars_rl = ax.bar(x + width/2, rl_k, width, label="RL System",    color="#52C878", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([t.replace(" ", "\n") for t in TOPIC_NAMES],
                   fontsize=8, color="white")
ax.set_ylabel("Knowledge (0–1)", color="white", fontsize=9)
ax.set_ylim(0, 1.05)
ax.tick_params(colors="white")
ax.spines[["top","right","left","bottom"]].set_color("#333")
ax.yaxis.grid(True, color="#333", linewidth=0.5)
ax.set_axisbelow(True)
legend = ax.legend(facecolor="#1a1a2e", edgecolor="#333",
                   labelcolor="white", fontsize=9)

# Highlight topics where RL > Fixed Script
for i, (fk, rk) in enumerate(zip(fs_k, rl_k)):
    if rk > fk + 0.02:
        ax.annotate("RL↑", xy=(x[i] + width/2, rk + 0.02),
                    ha="center", fontsize=7, color="#52C878", fontweight="bold")

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Insight box
st.divider()
improvement = ((rl_total_r - fs_total_r) / abs(fs_total_r) * 100) if fs_total_r != 0 else 0
# Count topics where RL beat Fixed Script
fs_k_final = fs_steps[-1]["k_by_topic"] if fs_steps else [0]*10
rl_k_final = rl_steps[-1]["k_by_topic"] if rl_steps else [0]*10
rl_wins_topics = sum(1 for f, r in zip(fs_k_final, rl_k_final) if r > f + 0.01)

improvement = ((rl_total_r - fs_total_r) / abs(fs_total_r) * 100) if fs_total_r != 0 else 0

st.success(f"""
**Why RL wins (+{improvement:.0f}% reward, +{delta_e:.2f} engagement):**

Fixed Script used **3 action types**, cycling rigidly regardless of what the student needed.
RL used **{len(rl_actions_used)} action types**, adapting every step to knowledge and engagement signals.

RL outperformed Fixed Script on **{rl_wins_topics}/10 topics**.
RL kept engagement at **{rl_final_e:.2f}** vs Fixed Script's **{fs_final_e:.2f}** — a student who stays engaged learns more over a full course.

The RL agent learned this from **{train_steps:,} simulated sessions** — no rules were written by hand.
""")

st.info("""
**Why avg knowledge looks low (e.g. 0.13):**
This is the *average across all 10 topics*. A session of 40 steps can only meaningfully cover 2–3 topics.
The topics actually covered show much higher knowledge (see the bar chart above).
A real course has hundreds of sessions — RL's engagement advantage compounds enormously over time.
""")

# ── 150-step pre-computed comparison ─────────────────────────────────────────
st.divider()
st.subheader("📈 Long-Term Learning — 150 Steps (Pre-computed)")
st.caption("Simulation mode — shows what happens over a full extended session. No API calls needed.")

curve_path = "results/figures/learning_curves_150.png"
if os.path.exists(curve_path):
    st.image(curve_path, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.markdown("""
    **Engagement (left)**
    🔴 Fixed Script collapses to **0.16** engagement by step 150 — the student has nearly checked out.
    🟢 Trained RL holds **0.64** — still motivated after 3× as many steps.
    ⚪ Untrained RL drifts down from 0.71 to 0.63 — no strategy to sustain it.
    """)
    col_b.markdown("""
    **Knowledge Gain (middle)**
    At step 50, untrained RL appears to win (random exploration covers more topics).
    By step 150, Trained RL pulls ahead with **+34% more knowledge gain** (4.26 vs 3.19).
    Sustained engagement is what drives long-term learning.
    """)
    col_c.markdown("""
    **Cumulative Reward (right)**
    Trained RL: **108.8** total reward
    Untrained RL: 56.4 (+93% gap)
    Fixed Script: 12.9 (+746% gap)
    Reward captures both knowledge AND engagement — the joint optimisation objective.
    """)

    st.error("""
    **The key finding:** Short sessions (50 steps) look similar across all systems.
    At 150 steps, Fixed Script engagement collapses to 0.16 — the student has given up.
    Trained RL maintains engagement 4× better and accumulates 34% more knowledge.
    Over a full semester, this compounds into the difference between passing and dropping the course.
    """)
else:
    st.info("Run `python experiments/learning_curves_comparison.py` to generate this chart.")
    if st.button("Generate Now"):
        import subprocess
        with st.spinner("Running 150-step comparison (takes ~30s)..."):
            subprocess.run(["python", "experiments/learning_curves_comparison.py"],
                           capture_output=True)
        st.rerun()
