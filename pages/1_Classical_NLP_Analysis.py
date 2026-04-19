import streamlit as st
import pandas as pd
from preprocessing import execute_preprocessing_pipeline, prepare_text_for_summary
from modeling import *
from utils import (
    generate_extractive_summary,
    render_silhouette_chart,
    render_similarity_heatmap,
    load_corpus_from_directory,
    process_uploaded_files,
)
import plotly.express as px
import plotly.graph_objects as go
import os
import re

# ---------------------------------------------------------------------------
# Custom CSS — Premium Dark Theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ---- Import Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---- Global (scoped to avoid breaking Streamlit icon fonts) ---- */
html, body, p, h1, h2, h3, h4, h5, h6, span, div, li, td, th, label, input, textarea, button {
    font-family: 'Inter', sans-serif;
}
/* Restore Material Symbols for Streamlit icon buttons */
[data-testid="collapsedControl"] * ,
.material-symbols-rounded {
    font-family: 'Material Symbols Rounded' !important;
}

/* ---- Hero Banner ---- */
.hero-banner {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 40%, #2c5364 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.hero-banner h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #e0e0e0, #a8edea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-banner p {
    margin: 0.5rem 0 0 0;
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 300;
}

/* ---- Stat Cards ---- */
.stat-card {
    background: linear-gradient(145deg, #1a1f2e, #151923);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.4);
}
.stat-card .stat-value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a8edea, #fed6e3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-card .stat-label {
    font-size: 0.82rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 0.35rem;
}

/* ---- Keyword Pill ---- */
.keyword-pill {
    display: inline-block;
    background: rgba(168, 237, 234, 0.08);
    border: 1px solid rgba(168, 237, 234, 0.25);
    color: #a8edea;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 4px 4px 4px 0;
    transition: background 0.2s;
}
.keyword-pill:hover {
    background: rgba(168, 237, 234, 0.15);
}

/* ---- Cluster Card ---- */
.cluster-card {
    background: linear-gradient(145deg, #1e2433, #171c28);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}
.cluster-card h3 {
    margin-top: 0;
    font-weight: 600;
    color: #e2e8f0;
}
.cluster-label {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ---- Summary Item ---- */
.summary-item {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #667eea;
    padding: 0.6rem 1rem;
    margin: 0.5rem 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #cbd5e1;
}

/* ---- Section Divider ---- */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 1.5rem 0;
}

/* ---- Tab Styling ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 24px;
    font-weight: 500;
}

/* ---- Expander + Info boxes ---- */
div[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.015) !important;
}

/* ---- Modal Document Viewer ---- */
.doc-viewer {
    max-height: 500px;
    overflow-y: auto;
    padding: 1.2rem;
    border-radius: 10px;
    background: #0d1117;
    border: 1px solid rgba(255,255,255,0.06);
    line-height: 1.7;
    font-size: 15px;
    white-space: pre-wrap;
    color: #c9d1d9;
}

/* ---- Legend badges ---- */
.legend-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.82rem;
    margin-right: 8px;
}
.legend-summary { background-color: #2ea043; color: white; }
.legend-keyword { background-color: #da3633; color: white; }

/* ---- LDA Topic Tag ---- */
.topic-tag {
    display: inline-block;
    background: rgba(80, 200, 120, 0.08);
    border: 1px solid rgba(80, 200, 120, 0.3);
    color: #50C878;
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 0.88rem;
    font-weight: 500;
    line-height: 1.5;
}

/* ---- Hide default Streamlit padding around metrics ---- */
div[data-testid="stMetric"] {
    display: none;
}
</style>
"""

# ---------------------------------------------------------------------------
# Plotly Theme
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#94a3b8'),
    title_font=dict(size=16, color='#e2e8f0'),
    margin=dict(t=50, b=40, l=50, r=30),
)

# ---------------------------------------------------------------------------
# Document Viewer Modal
# ---------------------------------------------------------------------------

def _highlight_text(text, keywords, summary_sentences):
    for sent in summary_sentences:
        pattern = re.escape(sent.strip())
        text = re.sub(
            pattern,
            f"<mark style='background-color:#2ea043; color:white; padding: 1px 3px; border-radius: 3px;'>{sent}</mark>",
            text,
            flags=re.IGNORECASE
        )
    for word in keywords:
        pattern = r"\b" + re.escape(word) + r"\b"
        text = re.sub(
            pattern,
            f"<span style='background-color:#da3633; color:white; padding: 1px 3px; border-radius: 3px;'>{word}</span>",
            text,
            flags=re.IGNORECASE
        )
    return text


@st.dialog("Detailed Analysis Report", width="large")
def show_document_modal(doc_name, raw_text, cleaned_text, keywords, summary_sentences):
    st.markdown(
        f"### {doc_name}\n\n"
        "<span class='legend-badge legend-summary'>Summary Sentence</span>"
        "<span class='legend-badge legend-keyword'>Keyword</span>"
        "<hr style='margin: 12px 0; border-color: rgba(255,255,255,0.06);'>",
        unsafe_allow_html=True
    )

    view_highlighted, view_original = st.tabs(["Highlighted Analysis", "Original Text"])

    with view_highlighted:
        highlighted = _highlight_text(cleaned_text, keywords, summary_sentences)
        st.markdown(f"<div class='doc-viewer'>{highlighted}</div>", unsafe_allow_html=True)

    with view_original:
        st.markdown(f"<div class='doc-viewer'>{raw_text}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page Config & CSS Injection
# ---------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="NLP Research Analyzer")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Hero Banner
st.markdown(
    """
    <div class='hero-banner'>
        <h1>NLP Research Analyzer</h1>
        <p>Uncover hidden themes, measure document similarity, and cluster research papers using classical NLP techniques.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("## Configuration")
st.sidebar.caption("Fine-tune the analysis pipeline and choose your data source.")

vectorization_mode = st.sidebar.radio(
    "Vectorization Mode",
    ["TF-IDF (Classical)", "Semantic Embeddings (SBERT)"],
    index=0,
    help="TF-IDF uses lexical word frequencies. Semantic Embeddings use a pre-trained encoder to capture meaning."
)
use_semantic = vectorization_mode == "Semantic Embeddings (SBERT)"

preserve_numbers = st.sidebar.toggle(
    "Retain Numerical Data (e.g., statistics, years)",
    value=True,
    disabled=use_semantic
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Source")

CORPUS_OPTIONS = {
    "Upload My Own Files": None,
    "Primary Research Corpus (Text)": "research_documents",
    "AI Research Papers (PDF)": "research_documents/pdf_papers",
    "Semantic Limitation Demo": "research_documents/semantic_demo",
}

selected_corpus_label = st.sidebar.selectbox(
    "Choose Corpus",
    options=list(CORPUS_OPTIONS.keys()),
    label_visibility="collapsed"
)

selected_corpus_path = CORPUS_OPTIONS[selected_corpus_label]

if selected_corpus_label == "Semantic Limitation Demo":
    st.sidebar.warning(
        "**Why this demo?**\n\n"
        "Three documents describe the *exact same event* — online shopping — "
        "using completely different vocabulary.\n\n"
        "TF-IDF is lexical, so it **cannot** recognise semantic equivalence. "
        "Expect near-zero similarity."
    )
elif selected_corpus_label == "AI Research Papers (PDF)":
    st.sidebar.info(
        "**PDF Corpus**\n\n"
        "Classic CS papers (Attention, BERT, ResNet, MapReduce, GFS) + an outlier (Cricket Rule Book). "
        "Perfect for demonstrating domain-based clustering."
    )

uploaded_files = st.sidebar.file_uploader(
    "Upload .txt or .pdf files",
    accept_multiple_files=True,
    disabled=(selected_corpus_path is not None)
)

# ---------------------------------------------------------------------------
# Load Documents
# ---------------------------------------------------------------------------

raw_docs = []
filenames = []

if selected_corpus_path is not None:
    raw_docs, filenames = load_corpus_from_directory(selected_corpus_path)
    if not raw_docs:
        st.error(f"No documents found in `{selected_corpus_path}/`.")
elif uploaded_files:
    raw_docs, filenames = process_uploaded_files(uploaded_files)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@st.cache_data
def run_tfidf_pipeline(raw_docs, preserve_numbers):
    processed_docs = [
        execute_preprocessing_pipeline(doc, preserve_numeric=preserve_numbers)
        for doc in raw_docs
    ]
    X, vectorizer = extract_tfidf_features(processed_docs)
    return processed_docs, X, vectorizer

@st.cache_data
def run_semantic_pipeline(raw_docs):
    X_sem = compute_semantic_embeddings(raw_docs)
    return X_sem


if raw_docs:
    # Always run TF-IDF for keywords, summarization, and LDA
    with st.spinner("Processing documents…"):
        processed_docs, X_tfidf, vectorizer = run_tfidf_pipeline(raw_docs, preserve_numbers)

    if use_semantic:
        with st.spinner("Computing semantic embeddings (SBERT)…"):
            X = run_semantic_pipeline(raw_docs)
        feature_count = X.shape[1]
        feature_label = "Embedding Dims"
        engine_label = "SBERT + K-Means"
    else:
        X = X_tfidf
        feature_count = len(vectorizer.get_feature_names_out())
        feature_label = "Vocabulary Terms"
        engine_label = "K-Means++"

    # -- Stat Cards --
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='stat-card'><div class='stat-value'>{len(raw_docs)}</div>"
            "<div class='stat-label'>Documents Loaded</div></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div class='stat-card'><div class='stat-value'>{feature_count}</div>"
            f"<div class='stat-label'>{feature_label}</div></div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div class='stat-card'><div class='stat-value'>{engine_label}</div>"
            "<div class='stat-label'>Clustering Engine</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Clustering Insights", "LDA Topic Modeling", "Similarity Matrix"])

    # ---------------------------------------------------------------
    # TAB 3 — Similarity Matrix
    # ---------------------------------------------------------------
    with tab3:
        st.subheader("Document Similarity (Cosine Matrix)")
        with st.expander("How does Cosine Similarity work?"):
            st.write(
                "Cosine similarity measures the angle between two document vectors in TF-IDF space. "
                "**1.0** = identical term distributions, **0.0** = nothing in common. "
                "It's length-agnostic — a 10-page and 2-page paper on the same topic still score highly."
            )

        if len(raw_docs) >= 2:
            with st.spinner("Computing pairwise similarities…"):
                similarity = calculate_cosine_similarity(X)
                fig_sim = render_similarity_heatmap(similarity, filenames)
                fig_sim.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_sim, width='stretch')
        else:
            st.warning("Upload at least 2 documents to view similarity.")

    # ---------------------------------------------------------------
    # TAB 2 — LDA
    # ---------------------------------------------------------------
    with tab2:
        st.subheader("Topic Extraction via LDA")
        with st.expander("What is LDA?"):
            st.write(
                "Latent Dirichlet Allocation discovers hidden 'topics' — each topic is a distribution "
                "over words, and each document is a mixture of topics. Great for finding dominant themes "
                "without manual labelling."
            )

        if len(raw_docs) >= 2:
            n_topics = st.slider("Number of Topics", min_value=2, max_value=min(6, len(raw_docs)), value=3, key="lda_slider")

            with st.spinner("Fitting LDA model…"):
                lda_model = perform_lda_modeling(X_tfidf, n_topics=n_topics)

            feature_names = vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(lda_model.components_):
                top_features_ind = topic.argsort()[:-10 - 1:-1]
                top_features = [feature_names[i] for i in top_features_ind]
                topic_html = f"<span class='topic-tag'>{', '.join(top_features)}</span>"
                st.markdown(f"**Theme {topic_idx + 1}:** &nbsp; {topic_html}", unsafe_allow_html=True)
                st.markdown("")
        else:
            st.warning("Upload at least 2 documents for topic modeling.")

    # ---------------------------------------------------------------
    # TAB 1 — Clustering
    # ---------------------------------------------------------------
    with tab1:
        st.subheader("Semantic K-Means Clustering")
        with st.expander("How does K-Means Clustering work?"):
            st.write(
                "K-Means partitions documents into *k* groups by minimising distance to cluster centroids. "
                "We use the **Silhouette Score** (−1 to 1) to auto-suggest the best *k* — higher = better separation."
            )

        if len(raw_docs) >= 2:
            with st.spinner("Evaluating optimal cluster count…"):
                suggested_k, scores_per_k = calculate_optimal_clusters(X)

            # -- Silhouette Chart --
            if len(raw_docs) >= 3 and scores_per_k:
                st.markdown("#### Silhouette Score Evaluation")
                fig_sil = render_silhouette_chart(scores_per_k)
                if fig_sil:
                    fig_sil.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig_sil, width='stretch')
                    st.caption("Higher silhouette score → better-separated clusters. The peak indicates the recommended *k*.")
            elif len(raw_docs) < 3:
                st.info("Need at least 3 documents for silhouette analysis.")

            k = st.slider(
                "Number of Clusters (K-Means)",
                min_value=1,
                max_value=len(raw_docs),
                value=suggested_k
            )

            if k > 0:
                with st.spinner("Running K-Means…"):
                    labels = perform_kmeans_clustering(X, k)
                cluster_df = pd.DataFrame({"Document": filenames, "Cluster": labels})

                if k > 1:
                    coords = apply_dimensionality_reduction(X, n_components=2)
                    cluster_df['PCA1'] = coords[:, 0]
                    cluster_df['PCA2'] = coords[:, 1]
                    cluster_df['Cluster'] = cluster_df['Cluster'].astype(str)

                    fig = px.scatter(
                        cluster_df, x='PCA1', y='PCA2', color='Cluster',
                        hover_name='Document', title="Semantic Clustering (2D PCA)",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig.update_traces(marker=dict(size=18, opacity=0.85, line=dict(width=1.5, color='rgba(255,255,255,0.2)')))
                    fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("K=1 — All documents in a single cluster. PCA scatter disabled.")

                # -- Cluster-level features --
                cluster_texts = {i: "" for i in range(k)}
                for label, text in zip(labels, raw_docs):
                    cluster_texts[label] += text + " "

                cluster_list = [cluster_texts[i] for i in range(k)]

                with st.spinner("Extracting cluster insights…"):
                    processed_clusters = [execute_preprocessing_pipeline(c, preserve_numeric=preserve_numbers) for c in cluster_list]
                    cluster_X, cluster_vectorizer = extract_tfidf_features(processed_clusters)

                    cluster_vocab_size = len(cluster_vectorizer.get_feature_names_out())
                    dynamic_top_n = max(3, min(10, int(0.1 * cluster_vocab_size)))

                    cluster_keywords = identify_top_keywords(
                        cluster_vectorizer, cluster_X, top_n=dynamic_top_n
                    )

                    # Per-document summaries within each cluster
                    cluster_doc_summaries = {}
                    for cid in range(k):
                        doc_indices = [i for i, lbl in enumerate(labels) if lbl == cid]
                        per_doc = []
                        for idx in doc_indices:
                            readable = prepare_text_for_summary(raw_docs[idx], preserve_numeric=preserve_numbers)
                            sents = generate_extractive_summary(readable, cluster_vectorizer, top_n=2)
                            per_doc.append((filenames[idx], sents))
                        cluster_doc_summaries[cid] = per_doc

                st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                st.subheader("Cluster Insights")

                for cluster_id in range(k):
                    st.markdown(f"<div class='cluster-card'>", unsafe_allow_html=True)
                    st.markdown(f"<span class='cluster-label'>Cluster {cluster_id}</span>", unsafe_allow_html=True)

                    # Keywords as pills
                    pills_html = "".join(
                        f"<span class='keyword-pill'>{kw}</span>" for kw in cluster_keywords[cluster_id]
                    )
                    st.markdown(f"**Keywords**<br>{pills_html}", unsafe_allow_html=True)

                    # Per-document summary — uniform styling
                    st.markdown("**Summary**", unsafe_allow_html=True)
                    for doc_name, sents in cluster_doc_summaries[cluster_id]:
                        if sents:
                            combined = " ".join(sents)
                            st.markdown(
                                f"<div class='summary-item'><strong>{doc_name}:</strong> {combined}</div>",
                                unsafe_allow_html=True
                            )

                    # Document buttons
                    st.markdown("**Documents**")
                    cluster_docs_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                    for idx in cluster_docs_indices:
                        doc_name = filenames[idx]
                        if st.button(doc_name, key=f"btn_cluster_{cluster_id}_{idx}", use_container_width=True):
                            cleaned_doc = prepare_text_for_summary(raw_docs[idx], preserve_numeric=preserve_numbers)
                            doc_sents = [s for dn, ss in cluster_doc_summaries[cluster_id] if dn == doc_name for s in ss]
                            show_document_modal(
                                doc_name, raw_docs[idx], cleaned_doc,
                                cluster_keywords[cluster_id], doc_sents
                            )

                    st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.subheader("All Documents")
                global_keywords = identify_top_keywords(vectorizer, X, top_n=8)
                readable_docs = [prepare_text_for_summary(doc, preserve_numeric=preserve_numbers) for doc in raw_docs]
                global_summaries = [generate_extractive_summary(doc, vectorizer, top_n=4) for doc in readable_docs]
                for idx, doc_name in enumerate(filenames):
                    if st.button(doc_name, key=f"btn_all_{idx}"):
                        show_document_modal(doc_name, raw_docs[idx], readable_docs[idx], global_keywords[idx], global_summaries[idx])
        else:
            st.warning("Upload at least 2 documents to view clustering.")

            st.subheader("All Documents")
            global_keywords = identify_top_keywords(vectorizer, X, top_n=8)
            readable_docs = [prepare_text_for_summary(doc, preserve_numeric=preserve_numbers) for doc in raw_docs]
            global_summaries = [generate_extractive_summary(doc, vectorizer, top_n=4) for doc in readable_docs]
            for idx, doc_name in enumerate(filenames):
                if st.button(doc_name, key=f"btn_single_{idx}"):
                    show_document_modal(doc_name, raw_docs[idx], readable_docs[idx], global_keywords[idx], global_summaries[idx])
else:
    # Empty state
    st.markdown(
        """
        <div style='text-align:center; padding: 4rem 2rem; color: #64748b;'>
            <h3 style='color: #94a3b8; font-weight: 500;'>No documents loaded</h3>
            <p>Select a sample corpus from the sidebar or upload your own files to get started.</p>
        </div>
        """,
        unsafe_allow_html=True
    )