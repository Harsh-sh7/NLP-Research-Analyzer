import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import plotly.express as px
import pandas as pd
import os
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Extractive Summarization (TextRank)
# ---------------------------------------------------------------------------

def generate_extractive_summary(text, vectorizer, top_n=3):
    """TextRank-based extractive summarizer with guard-rails for large PDFs."""
    # Cap input length to avoid extreme processing times
    MAX_INPUT_CHARS = 50_000
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    raw_sentences = nltk.sent_tokenize(text)

    # Filter: keep sentences between 20-500 chars (removes TOC junk,
    # footer fragments, but keeps legitimately long research sentences)
    MIN_SENT_LEN = 20
    MAX_SENT_LEN = 500
    sentences = [
        s.strip() for s in raw_sentences
        if MIN_SENT_LEN < len(s.strip()) <= MAX_SENT_LEN
    ]

    if not sentences:
        return []

    sentence_vectors = vectorizer.transform(sentences)
    sim_matrix = cosine_similarity(sentence_vectors)

    # TextRank: PageRank over sentence similarity graph
    nx_graph = nx.from_numpy_array(sim_matrix)

    try:
        scores = nx.pagerank(nx_graph)
    except Exception:
        scores = {i: sim_matrix.sum(axis=1)[i] for i in range(len(sentences))}

    n_sentences = min(top_n, len(sentences))

    ranked_scored = sorted(
        ((scores[i], i) for i in range(len(sentences))), reverse=True
    )
    top_indices = [idx for _, idx in ranked_scored[:n_sentences]]

    # Chronological order for readability
    chronological_indices = sorted(top_indices)

    summary = []
    for i in chronological_indices:
        sent = sentences[i].strip()
        if not sent.endswith(('.', '!', '?', '"', "'")):
            sent += "."
        summary.append(sent)

    return summary

# ---------------------------------------------------------------------------
# Silhouette Score Visualization
# ---------------------------------------------------------------------------

def render_silhouette_chart(scores_per_k):
    """Build a Plotly line chart showing silhouette scores across k values."""
    if not scores_per_k:
        return None
    
    valid = {k: v for k, v in scores_per_k.items() if v >= -1}
    if not valid:
        return None
    
    df = pd.DataFrame(
        list(valid.items()),
        columns=["Clusters (k)", "Silhouette Score"]
    )
    
    fig = px.line(
        df,
        x="Clusters (k)",
        y="Silhouette Score",
        markers=True,
        title="Cluster Quality — Silhouette Score Analysis"
    )
    fig.update_traces(
        line=dict(color='#a8edea', width=3),
        marker=dict(size=10, color='#667eea', line=dict(width=2, color='#a8edea'))
    )
    
    score_min = df["Silhouette Score"].min()
    score_max = df["Silhouette Score"].max()
    padding = 0.05
    fig.update_yaxes(range=[score_min - padding, score_max + padding])
    fig.update_layout(xaxis=dict(dtick=1))
    
    return fig

# ---------------------------------------------------------------------------
# Improved Similarity Heatmap
# ---------------------------------------------------------------------------

def render_similarity_heatmap(sim_matrix, filenames):
    """Create a Plotly heatmap with truncated axis labels and full-name hover."""
    # Shorten names for axes — max 22 chars
    short_names = [
        (name[:20] + "…") if len(name) > 22 else name
        for name in filenames
    ]
    
    sim_df = pd.DataFrame(sim_matrix, index=short_names, columns=short_names)
    
    fig = px.imshow(
        sim_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Viridis"
    )
    
    # Full filename grid for hover tooltips
    hover_grid = [
        [f"Row: {fy}<br>Col: {fx}" for fx in filenames]
        for fy in filenames
    ]
    fig.update_traces(
        customdata=hover_grid,
        hovertemplate="%{customdata}<br>Similarity: %{z:.2f}<extra></extra>"
    )
    
    return fig

# ---------------------------------------------------------------------------
# Corpus Loading Utilities
# ---------------------------------------------------------------------------

def load_corpus_from_directory(directory_path):
    """Read all .txt and .pdf files from the given directory.
    Returns (raw_docs, filenames) lists."""
    raw_docs = []
    filenames = []
    
    if not os.path.exists(directory_path):
        return raw_docs, filenames
    
    for fname in sorted(os.listdir(directory_path)):
        fpath = os.path.join(directory_path, fname)
        
        if fname.lower().endswith(".pdf"):
            try:
                pdf = PdfReader(fpath)
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                raw_docs.append(text)
                filenames.append(fname)
            except Exception:
                pass
        elif fname.lower().endswith(".txt"):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    raw_docs.append(f.read())
                filenames.append(fname)
            except Exception:
                pass
    
    return raw_docs, filenames

def process_uploaded_files(uploaded_files):
    """Process Streamlit UploadedFile objects, extracting text from each."""
    raw_docs = []
    filenames = []
    
    for file in uploaded_files:
        if file.name.lower().endswith(".pdf"):
            try:
                pdf = PdfReader(file)
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                raw_docs.append(text)
                filenames.append(file.name)
            except Exception:
                pass
        elif file.name.lower().endswith(".txt"):
            text = file.getvalue().decode("utf-8", errors="ignore")
            raw_docs.append(text)
            filenames.append(file.name)
    
    return raw_docs, filenames