import streamlit as st

st.set_page_config(
    page_title="Unified Research Platform",
    page_icon="🔬",
    layout="wide",
)

# Custom CSS for Landing Page
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.hero {
    background: linear-gradient(135deg, #09090b 0%, #1a1f2e 100%);
    padding: 4rem 2rem;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
}

.hero h1 {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 1rem;
    background: linear-gradient(90deg, #e0e0e0, #a8edea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    font-size: 1.25rem;
    color: #94a3b8;
    max-width: 800px;
    margin: 0 auto;
    font-weight: 300;
    line-height: 1.6;
}

.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-top: 2rem;
}

.feature-card {
    background: #11151e;
    border: 1px solid rgba(255,255,255,0.06);
    padding: 2.5rem;
    border-radius: 12px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    border-color: rgba(102, 126, 234, 0.4);
}

.feature-card h2 {
    color: #f1f5f9;
    font-size: 1.8rem;
    margin-bottom: 1rem;
    font-weight: 600;
}

.feature-card h2 span {
    font-size: 2rem;
    margin-right: 0.5rem;
    vertical-align: middle;
}

.feature-card p {
    color: #94a3b8;
    font-size: 1.05rem;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}

.feature-tag {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: #cdd6f4;
    padding: 0.4rem 0.8rem;
    border-radius: 6px;
    font-size: 0.85rem;
    margin: 0.25rem 0.25rem 0 0;
    font-weight: 500;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Unified Research Platform</h1>
    <p>A comprehensive ecosystem combining classical Natural Language Processing algorithms with state-of-the-art Agentic Generative AI for end-to-end research synthesis.</p>
</div>

<div class="feature-grid">
    <div class="feature-card">
        <h2><span>🔍</span> Classical NLP Analysis</h2>
        <p>Uncover hidden themes, measure text similarity, and cluster your private research corpus using TF-IDF, Semantic Embeddings (SBERT), K-Means, and Latent Dirichlet Allocation.</p>
        <div>
            <span class="feature-tag">Clustering</span>
            <span class="feature-tag">Topic Modeling</span>
            <span class="feature-tag">Extractive Summarization</span>
        </div>
        <p style="margin-top: 1.5rem; font-size: 0.9rem; color: #64748b;">👈 Select <b>Classical NLP Analysis</b> in the sidebar to begin.</p>
    </div>
    <div class="feature-card">
        <h2><span>🤖</span> Agentic GenAI Research</h2>
        <p>Deploy an autonomous swarm of AI agents to decompose queries, scrape the web, extract facts into a vector database, and synthesize fully cited, structured research reports.</p>
        <div>
            <span class="feature-tag">LangGraph Workflow</span>
            <span class="feature-tag">Web Scraping</span>
            <span class="feature-tag">RAG / FAISS</span>
            <span class="feature-tag">Automated Drafting</span>
        </div>
        <p style="margin-top: 1.5rem; font-size: 0.9rem; color: #64748b;">👈 Select <b>Agentic GenAI Research</b> in the sidebar to begin.</p>
    </div>
</div>
""", unsafe_allow_html=True)
