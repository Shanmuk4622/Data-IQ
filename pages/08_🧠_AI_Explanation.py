"""pages/08_🧠_AI_Explanation.py — Groq AI narrative report."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("🧠 AI Explanation")
st.caption("Powered by Groq AI Models (LLaMA-3.3 70B with automatic fallbacks) — reads your dataset analysis and writes a professional report")

df = st.session_state.get("df_raw")
profile = st.session_state.get("profile")

if df is None or profile is None:
    st.info("👈 Upload a dataset or try a demo from the sidebar.")
    st.stop()

groq_key = st.session_state.get("groq_api_key", "")
health = st.session_state.get("health_score") or profile.get("health_score", {})

# ── API Key check ─────────────────────────────────────────────────────────────
if not groq_key:
    st.markdown(
        """<div class="warning-box">
        ⚠️ <strong>Groq API key required</strong><br>
        Enter your key in the sidebar. Get a free key at 
        <a href="https://console.groq.com" target="_blank" style="color:#8B5CF6;">console.groq.com</a>
        — inference is free with generous rate limits.
        </div>""",
        unsafe_allow_html=True,
    )

# ── Generate Report Button ────────────────────────────────────────────────────
existing_report = st.session_state.get("groq_report")

c1, c2 = st.columns([1, 3])
with c1:
    generate_btn = st.button(
        "🧠 Generate AI Report",
        type="primary",
        disabled=not groq_key,
        use_container_width=True,
    )
with c2:
    if existing_report:
        regen_btn = st.button("🔄 Regenerate", use_container_width=False)
    else:
        regen_btn = False

if generate_btn or regen_btn:
    if not groq_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        from modules.groq_client import build_profile_summary, get_ai_explanation
        with st.spinner("🧠 Groq AI is analysing your dataset (using llama-3.3-70b-versatile or fallback)…"):
            summary = build_profile_summary(profile, health)
            report = get_ai_explanation(summary, groq_key)
            st.session_state["groq_report"] = report
        st.success("AI report generated!")
        st.rerun()

# ── Display Report ────────────────────────────────────────────────────────────
if existing_report:
    st.divider()
    st.markdown(
        f'<div class="ai-report">{existing_report}</div>',
        unsafe_allow_html=True,
    )
    
    # Download options
    st.divider()
    st.download_button(
        "📥 Download AI Report (.md)",
        data=existing_report.encode("utf-8"),
        file_name="ai_analysis_report.md",
        mime="text/markdown",
    )

# ── Quick Question ────────────────────────────────────────────────────────────
st.divider()
st.markdown("## 💬 Ask a Quick Question")
st.caption("Ask anything about this dataset — uses a faster model for quick answers")

quick_q = st.text_input(
    "Your question:",
    placeholder="e.g. Which feature is most important? Is this dataset suitable for neural networks?",
)

if st.button("Ask", disabled=not (groq_key and quick_q)):
    from modules.groq_client import build_profile_summary, get_quick_insight
    summary = build_profile_summary(profile, health)
    with st.spinner("Thinking…"):
        answer = get_quick_insight(quick_q, summary, groq_key)
    st.markdown(f'<div class="info-box">{answer}</div>', unsafe_allow_html=True)
