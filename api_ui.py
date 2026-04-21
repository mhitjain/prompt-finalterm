"""
API Demo UI — shows the FastAPI backend working through a simple frontend.

This is what a university LMS or mobile app would look like when
built on top of the Dewey RL Tutoring API.

Run (API server must be running first):
    uvicorn api_server:app --port 8000
    streamlit run api_ui.py --server.port 8503
"""

import streamlit as st
import requests
import json

API = "http://localhost:8000"

st.set_page_config(page_title="Dewey API Demo", page_icon="🔌", layout="wide")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center'>🔌 Dewey Tutoring API — Live Demo</h1>
<p style='text-align:center;color:#888'>
  Any app (mobile, web, LMS) can plug in here. This UI calls the REST API.
</p>
""", unsafe_allow_html=True)

# ── Check server health ───────────────────────────────────────────────────────
try:
    health = requests.get(f"{API}/health", timeout=2).json()
    st.success(f"✅ API server running — uptime {health['uptime_seconds']:.0f}s  |  "
               f"Active sessions: {health['active_sessions']}  |  "
               f"Version: {health['version']}")
except Exception:
    st.error("❌ API server not running. Start it with: `uvicorn api_server:app --port 8000`")
    st.stop()

st.divider()

# ── Two panels ────────────────────────────────────────────────────────────────
left, right = st.columns([1, 2])

with left:
    st.subheader("🎓 Start a Tutoring Session")

    subject = st.selectbox("Subject", ["calculus", "physics", "algorithms"])
    profile = st.selectbox("Student Profile",
                           [0, 1, 2, 3],
                           format_func=lambda x: ["Fast Learner","Slow Learner",
                                                   "Visual Learner","Practice Learner"][x])
    seed = st.number_input("Seed", value=42, step=1)

    if st.button("🚀 Create Session", type="primary", use_container_width=True):
        resp = requests.post(f"{API}/session/new",
                             json={"subject": subject, "profile_id": profile, "seed": int(seed)})
        if resp.ok:
            st.session_state["session_id"] = resp.json()["session_id"]
            st.session_state["history"]    = []
            st.session_state["step"]       = 0
            st.success(f"Session created: `{st.session_state['session_id']}`")
        else:
            st.error(f"Error: {resp.text}")

    st.divider()

    # Show raw JSON panel
    if st.session_state.get("session_id"):
        sid = st.session_state["session_id"]
        st.subheader("📡 What the API returns")
        st.caption("This is what your mobile app / LMS receives:")

        with st.expander("POST /session/new  (already called)", expanded=False):
            st.json({"session_id": sid,
                     "message": f"Session started for subject={subject}"})

        if st.session_state.get("last_step_json"):
            with st.expander("POST /session/{id}/step  (last step)", expanded=True):
                st.json(st.session_state["last_step_json"])

        status_resp = requests.get(f"{API}/session/{sid}/status")
        if status_resp.ok:
            with st.expander("GET /session/{id}/status", expanded=False):
                st.json(status_resp.json())

        metrics_resp = requests.get(f"{API}/metrics")
        if metrics_resp.ok:
            with st.expander("GET /metrics  (platform dashboard)", expanded=False):
                st.json(metrics_resp.json())

with right:
    st.subheader("📚 Tutoring Session")

    if not st.session_state.get("session_id"):
        st.info("👈 Create a session first.")
    else:
        sid = st.session_state["session_id"]

        # Knowledge progress bar
        status_resp = requests.get(f"{API}/session/{sid}/status")
        if status_resp.ok:
            s = status_resp.json()
            topics = ["Arithmetic","Algebra","Geometry","Statistics","Linear Alg",
                      "Calculus","Probability","Adv Stats","ML Basics","Deep Learning"]

            prog_cols = st.columns(5)
            for i, (t, k) in enumerate(zip(topics, s["knowledge_by_topic"])):
                with prog_cols[i % 5]:
                    st.metric(t[:8], f"{k:.0%}")

            ecol, scol, mcol = st.columns(3)
            ecol.metric("Engagement", f"{s['engagement']:.0%}")
            scol.metric("Step", s["step"])
            mcol.metric("Mastered", f"{s['n_mastered']}/10")

        st.divider()

        # Take step button
        if st.button("▶ Take Next Step", type="primary", use_container_width=True):
            step_resp = requests.post(f"{API}/session/{sid}/step",
                                      headers={"Content-Type": "application/json"},
                                      data="{}")
            if step_resp.ok:
                data = step_resp.json()
                st.session_state["last_step_json"] = data
                st.session_state["history"].append(data)
                st.session_state["step"] += 1
                st.rerun()
            else:
                st.error(f"Step failed: {step_resp.text}")

        # Chat-style transcript
        history = st.session_state.get("history", [])
        if history:
            st.markdown("**Session Transcript**")
            for entry in reversed(history[-8:]):  # show last 8
                action_color = {
                    "EXPLAIN_CONCEPT": "#4A90D9",
                    "SHOW_EXAMPLE":    "#7B68EE",
                    "ASK_EASY":        "#52C878",
                    "ASK_MEDIUM":      "#F0A500",
                    "ASK_HARD":        "#E05C5C",
                    "GIVE_HINT":       "#20B2AA",
                    "ENCOURAGE":       "#32CD32",
                }.get(entry["action"], "#888")

                st.markdown(
                    f"<div style='border-left:4px solid {action_color};"
                    f"padding:10px 14px;margin:6px 0;border-radius:4px;"
                    f"background:#111'>"
                    f"<span style='color:{action_color};font-size:11px;font-weight:700'>"
                    f"Step {entry['step']} · {entry['action']}</span><br>"
                    f"<span style='font-size:14px'>{entry['content']}</span><br>"
                    f"<span style='color:#888;font-size:11px'>"
                    f"reward {entry['reward']:+.2f} · "
                    f"knowledge {entry['mean_knowledge']:.3f} · "
                    f"engagement {entry['engagement']:.2f}"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )

            if history[-1].get("done"):
                st.balloons()
                st.success("🎓 Session complete!")

# ── Footer explanation ────────────────────────────────────────────────────────
st.divider()
st.markdown("""
**How this works:**
- This Streamlit page is just a *frontend* — it has no RL code itself
- Every button click calls the REST API running at `localhost:8000`
- Any app (React, mobile, Canvas LMS plugin) could replace this UI
- The API handles all the RL decisions, student state, and LLM calls
- **This is what production deployment looks like**
""")
