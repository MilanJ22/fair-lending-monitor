import streamlit as st
from src.agent import ask
from src.storage import query

st.set_page_config(
    page_title="Fair Lending Monitor",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default Streamlit header padding */
    .block-container { padding-top: 1.5rem; }

    /* App header */
    .app-header {
        background: linear-gradient(135deg, #0f2044 0%, #1a3a6b 100%);
        color: white;
        padding: 1.2rem 1.8rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        color: white;
        margin: 0;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #8eb4e8;
        margin: 0.25rem 0 0 0;
        font-size: 0.82rem;
    }

    /* Stat cards */
    .stat-card {
        background: #f7f9fc;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #1a3a6b;
    }
    .stat-card.red   { border-left-color: #dc3545; }
    .stat-card.yellow { border-left-color: #e6a817; }
    .stat-label {
        font-size: 0.68rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.1rem;
    }
    .stat-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f2044;
        line-height: 1.2;
    }

    /* Section labels */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #999;
        margin: 1.2rem 0 0.5rem 0;
    }

    /* Chat message tweaks */
    div[data-testid="stChatMessage"] { padding: 0.25rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Cached dataset stats ─────────────────────────────────────────────────────────
@st.cache_data
def load_stats():
    total_apps   = query("SELECT COUNT(*) AS n FROM applications").iloc[0]["n"]
    total_lenders = query("SELECT COUNT(DISTINCT lei) AS n FROM denial_rates").iloc[0]["n"]
    red_count    = query("SELECT COUNT(*) AS n FROM disparity_flags WHERE disparity_flag = 'Red'").iloc[0]["n"]
    yellow_count = query("SELECT COUNT(*) AS n FROM disparity_flags WHERE disparity_flag = 'Yellow'").iloc[0]["n"]
    return {
        "total_apps":    f"{total_apps:,}",
        "total_lenders": f"{total_lenders:,}",
        "red_count":     f"{red_count:,}",
        "yellow_count":  f"{yellow_count:,}",
    }


# ── Example queries ──────────────────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "Which MSA has the highest concentration of Red-flag disparity segments?",
    "Are the Red-flag lenders in Minneapolis depository banks or non-depository mortgage companies?",
    "Show me the individual flagged segments in Minneapolis for Black or African American applicants.",
    "Are there any lenders with a near-100% denial rate for Black or African American applicants?",
    "Show me the denial reasons for the highest-disparity lender in Minneapolis.",
    "Compare disparity ratios for independent mortgage companies vs. banks in the Denver MSA.",
]


# ── Session state ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='font-size:0.78rem; color:#666; margin-bottom:1rem;'>
            <strong style='color:#0f2044;'>2024 HMDA Data</strong><br>
            Home purchase loans · 15 major MSAs<br>
            Source: CFPB HMDA Data Browser
        </div>
    """, unsafe_allow_html=True)

    stats = load_stats()

    st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>Total Applications</div>
            <div class='stat-value'>{stats['total_apps']}</div>
        </div>
        <div class='stat-card'>
            <div class='stat-label'>Lenders Analyzed</div>
            <div class='stat-value'>{stats['total_lenders']}</div>
        </div>
        <div class='stat-card red'>
            <div class='stat-label'>Red Flag Segments (&ge;2.0x)</div>
            <div class='stat-value'>{stats['red_count']}</div>
        </div>
        <div class='stat-card yellow'>
            <div class='stat-label'>Yellow Flag Segments (&ge;1.5x)</div>
            <div class='stat-value'>{stats['yellow_count']}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Example Queries</div>", unsafe_allow_html=True)

    for q in EXAMPLE_QUERIES:
        if st.button(q, key=f"ex_{q[:30]}", use_container_width=True):
            st.session_state.pending_query = q

    st.markdown("---")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main area ────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class='app-header'>
        <h1>⚖ Fair Lending Compliance Monitor</h1>
        <p>
            Natural language analysis over 2024 HMDA data · 15 major MSAs ·
            Peer benchmarking · Disparity detection · Exam-ready recommendations
        </p>
    </div>
""", unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Resolve input — chat box takes priority over example button
user_input = st.chat_input("Ask a compliance question...")

if user_input is None and st.session_state.pending_query:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None

# Handle input
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent and stream response
    with st.chat_message("assistant"):
        with st.spinner("Querying dataset and generating recommendation..."):
            response = ask(user_input)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
