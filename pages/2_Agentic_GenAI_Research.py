import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Agentic Research",
    page_icon=":material/search:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* General Layout & Typography */
body {
    background-color: #09090b;
    color: #fafafa;
}
h1, h2, h3, h4, h5, h6 {
    color: #fafafa !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

/* Header */
.main-header {
    padding: 2rem 0 1rem;
    border-bottom: 1px solid #27272a;
    margin-bottom: 2rem;
}
.main-header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}
.main-subtitle {
    color: #a1a1aa;
    font-size: 1rem;
    font-weight: 400;
}

/* Sidebar */
.sidebar-header {
    padding-bottom: 1rem;
    border-bottom: 1px solid #27272a;
    margin-bottom: 1.5rem;
}
.sidebar-header h3 {
    font-size: 1.25rem;
    margin: 0;
}
.sidebar-section {
    margin-bottom: 1.5rem;
}
.sidebar-section h4 {
    font-size: 0.875rem;
    color: #a1a1aa;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}
.tech-tag {
    display: inline-block;
    background: #27272a;
    color: #e4e4e7;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.15rem;
    border: 1px solid #3f3f46;
}

/* Input Area */
div[data-testid="stTextArea"] textarea {
    border-radius: 6px !important;
    border: 1px solid #27272a !important;
    background: #09090b !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    color: #fafafa !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #fafafa !important;
    box-shadow: 0 0 0 1px #fafafa !important;
}

/* Primary Button */
.stButton > button[kind="primary"],
div.row-widget.stButton > button {
    background: #fafafa !important;
    color: #09090b !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    transition: background 0.2s ease !important;
}
.stButton > button:hover,
div.row-widget.stButton > button:hover {
    background: #e4e4e7 !important;
}

/* Download Button */
.stDownloadButton > button {
    background: #09090b !important;
    color: #fafafa !important;
    border: 1px solid #27272a !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover {
    background: #18181b !important;
    border-color: #3f3f46 !important;
}

/* Node Cards */
.node-card {
    padding: 1rem;
    border-radius: 6px;
    margin: 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.9rem;
    font-weight: 500;
    border: 1px solid #27272a;
}
.node-pending {
    background: #18181b;
    color: #a1a1aa;
}
.node-active {
    background: #09090b;
    border-color: #fafafa;
    color: #fafafa;
    box-shadow: 0 1px 3px rgba(0,0,0,0.5);
}
.node-done {
    background: #18181b;
    border-color: #3f3f46;
    color: #fafafa;
}
.node-icon { 
    font-size: 1.1rem; 
    color: #a1a1aa;
}
.node-active .node-icon {
    color: #fafafa;
}

/* Stats Cards */
.stats-card {
    background: #09090b;
    border: 1px solid #27272a;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
}
.stats-card .stat-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #fafafa;
}
.stats-card .stat-label {
    font-size: 0.75rem;
    color: #a1a1aa;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}

/* Report View */
.report-section {
    background: #09090b;
    border: 1px solid #27272a;
    border-radius: 8px;
    padding: 2rem;
    margin-top: 1rem;
}

/* Expander */
div[data-testid="stExpander"] {
    border: 1px solid #27272a !important;
    border-radius: 6px !important;
    background: #18181b !important;
}

/* Remove all Streamlit injected padding artifacts if needed */
.block-container {
    padding-top: 2rem !important;
}

.footer-area {
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid #27272a;
    text-align: center;
    color: #a1a1aa;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h3>Research Agent</h3>
        <div style="color: #a1a1aa; font-size: 0.875rem; margin-top: 0.25rem;">Autonomous Workflow</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>Architecture</h4>
        <div style="color: #d4d4d8; font-size: 0.875rem; line-height: 1.6;">
            • Planner <span style="color:#71717a;">(Decompose)</span><br>
            • Researcher <span style="color:#71717a;">(Search)</span><br>
            • Scraper <span style="color:#71717a;">(Extract)</span><br>
            • Synthesizer <span style="color:#71717a;">(Report)</span><br>
            • Reviewer <span style="color:#71717a;">(Validate)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>Tech Stack</h4>
        <span class="tech-tag">LangGraph</span>
        <span class="tech-tag">Groq</span>
        <span class="tech-tag">Tavily</span>
        <span class="tech-tag">FAISS</span>
        <span class="tech-tag">RAG</span>
    </div>
    """, unsafe_allow_html=True)

    groq_ok = bool(os.getenv("GROQ_API_KEY") and os.getenv("GROQ_API_KEY") != "your_groq_api_key_here")
    tavily_ok = bool(os.getenv("TAVILY_API_KEY") and os.getenv("TAVILY_API_KEY") != "tvly_your_tavily_api_key_here")
    
    st.markdown(f"""
    <div class="sidebar-section">
        <h4>API Status</h4>
        <div style="font-size: 0.875rem; color: #d4d4d8; display: flex; align-items: center; justify-content: space-between;">
            <span>Groq API</span>
            <span style="color: {'#10b981' if groq_ok else '#ef4444'};">{'Active' if groq_ok else 'Missing'}</span>
        </div>
        <div style="font-size: 0.875rem; color: #d4d4d8; display: flex; align-items: center; justify-content: space-between; margin-top:0.25rem;">
            <span>Tavily API</span>
            <span style="color: {'#10b981' if tavily_ok else '#ef4444'};">{'Active' if tavily_ok else 'Missing'}</span>
        </div>
        <div style="font-size: 0.875rem; color: #d4d4d8; display: flex; align-items: center; justify-content: space-between; margin-top:0.25rem;">
            <span>Embeddings</span>
            <span style="color: #10b981;">Local</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "history" in st.session_state and st.session_state.history:
        st.markdown("""<div class="sidebar-section"><h4>Recent Queries</h4></div>""", unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(f"<div style='color:#a1a1aa; font-size:0.875rem; padding:0.25rem 0;'>• {item[:45]}...</div>", unsafe_allow_html=True)


# ── Main Content ──
st.markdown("""
<div class="main-header">
    <h1>Research System</h1>
    <p class="main-subtitle">Compile comprehensive, verified reports using autonomous agents.</p>
</div>
""", unsafe_allow_html=True)

if "report" not in st.session_state:
    st.session_state.report = None
if "stats" not in st.session_state:
    st.session_state.stats = None
if "history" not in st.session_state:
    st.session_state.history = []

col_l, col_m, col_r = st.columns([0.2, 4, 0.2])
with col_m:
    query = st.text_area(
        "research_input",
        placeholder="Enter your research topic...",
        height=100,
        label_visibility="collapsed",
    )

    start = st.button("Start Research", use_container_width=True)

NODE_META = {
    "planner":     (":material/format_list_bulleted:", "Planner", "Decomposing query into specific sub-tasks"),
    "researcher":  (":material/travel_explore:", "Researcher", "Gathering data via Tavily search"),
    "scraper":     (":material/find_in_page:", "Scraper", "Extracting and indexing page contents"),
    "synthesizer": (":material/edit_document:", "Synthesizer", "Compiling findings into a report"),
    "reviewer":    (":material/fact_check:", "Reviewer", "Validating content and citations"),
}

NODE_ORDER = list(NODE_META.keys())

# ── Run Agent ──
if start and query.strip():
    if not groq_ok:
        st.error("Please set your GROQ_API_KEY in the .env file.")
        st.stop()
    if not tavily_ok:
        st.error("Please set your TAVILY_API_KEY in the .env file.")
        st.stop()

    st.session_state.report = None
    st.session_state.stats = None

    from agent.graph import build_graph
    agent = build_graph()

    initial_state = {
        "research_query": query.strip(),
        "task_list": [],
        "completed_tasks": [],
        "search_results": [],
        "scraped_urls": [],
        "context_library": [],
        "citations": {},
        "report_draft": "",
        "revision_count": 0,
        "reviewer_feedback": "",
        "status": "starting",
    }

    st.markdown("### Execution Graph")

    placeholders = {}
    for name in NODE_ORDER:
        placeholders[name] = st.empty()
        icon, label, desc = NODE_META[name]
        placeholders[name].markdown(
            f'<div class="node-card node-pending"><span style="display:none;">{icon}</span><span class="node-icon">○</span> <b>{label}</b> &nbsp;|&nbsp; {desc}</div>',
            unsafe_allow_html=True,
        )

    completed = set()
    final_state = {}
    iteration = 0

    for event in agent.stream(initial_state):
        for node_name, node_output in event.items():
            if node_name.startswith("__"):
                continue

            if node_name in NODE_META:
                icon, label, desc = NODE_META[node_name]
                placeholders[node_name].markdown(
                    f'<div class="node-card node-active"><span style="display:none;">{icon}</span><span class="node-icon">●</span> <b>{label}</b> &nbsp;|&nbsp; {desc}</div>',
                    unsafe_allow_html=True,
                )

            if isinstance(node_output, dict):
                final_state.update(node_output)

            if node_name in NODE_META:
                completed.add(node_name)
                icon, label, desc = NODE_META[node_name]
                placeholders[node_name].markdown(
                    f'<div class="node-card node-done"><span style="display:none;">{icon}</span><span class="node-icon">✓</span> <b>{label}</b> &nbsp;|&nbsp; Completed</div>',
                    unsafe_allow_html=True,
                )

            if node_name == "reviewer" and final_state.get("status") == "needs_revision":
                iteration += 1
                for n in NODE_ORDER[1:]:
                    if n != "reviewer":
                        icon2, label2, desc2 = NODE_META[n]
                        placeholders[n].markdown(
                            f'<div class="node-card node-pending"><span style="display:none;">{icon2}</span><span class="node-icon">○</span> <b>{label2}</b> &nbsp;|&nbsp; Revision {iteration}</div>',
                            unsafe_allow_html=True,
                        )

    report = final_state.get("report_draft", "")

    if report:
        st.session_state.report = report
        st.session_state.stats = {
            "tasks": len(final_state.get("task_list", [])),
            "sources": len(final_state.get("search_results", [])),
            "pages_scraped": len(final_state.get("scraped_urls", [])),
            "revisions": final_state.get("revision_count", 0),
            "context_chunks": len(final_state.get("context_library", [])),
        }
        st.session_state.history.append(query.strip())


# ── Display Report ──
if st.session_state.report:
    st.markdown("---")
    st.markdown("### Final Report")

    with st.container():
        st.markdown(f'<div class="report-section">', unsafe_allow_html=True)
        st.markdown(st.session_state.report)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
    with col_d2:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "Download Document",
            data=st.session_state.report,
            file_name=f"report_{timestamp}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if st.session_state.stats:
        st.markdown("---")
        st.markdown("### Execution Metrics")

        s = st.session_state.stats
        c1, c2, c3, c4, c5 = st.columns(5)

        for col, val, label in [
            (c1, s["tasks"], "Queries Generated"),
            (c2, s["sources"], "Search Results"),
            (c3, s["pages_scraped"], "Pages Processed"),
            (c4, s["context_chunks"], "RAG Vectors"),
            (c5, s["revisions"], "Review Cycles"),
        ]:
            with col:
                st.markdown(
                    f'<div class="stats-card"><div class="stat-value">{val}</div><div class="stat-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )

    with st.expander("View Raw State Output"):
        if st.session_state.stats:
            st.json({"query": st.session_state.history[-1] if st.session_state.history else "", "tasks": []})


st.markdown("""
<div class="footer-area">
    Built with LangGraph, Groq, Tavily, FAISS, and Streamlit
</div>
""", unsafe_allow_html=True)
