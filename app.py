import streamlit as st
import pandas as pd
from preprocessing import execute_preprocessing_pipeline
from modeling import *
from utils import generate_extractive_summary
from pypdf import PdfReader
import plotly.express as px
import os
import re

def highlight_text(text, keywords, summary_sentences):
    # Highlight summary sentences first (green)
    for sent in summary_sentences:
        pattern = re.escape(sent.strip())
        text = re.sub(
            pattern,
            f"<mark style='background-color:#bcf5bc; color:black'>{sent}</mark>",
            text,
            flags=re.IGNORECASE
        )

    # Highlight keywords (red)
    for word in keywords:
        pattern = r"\b" + re.escape(word) + r"\b"
        text = re.sub(
            pattern,
            f"<span style='background-color:#ff9999; color:black'>{word}</span>",
            text,
            flags=re.IGNORECASE
        )

    return text


@st.dialog("Detailed Analysis Report", width="large")
def show_document_modal(doc_name, doc_text, keywords, summary_sentences):
    st.markdown(
        f"### {doc_name}\n\n"
        "**Legend:** \n"
        "<span style='background-color:#bcf5bc; color:black; padding: 2px 6px; border-radius: 4px; font-weight: bold;'>Summary Sentence</span> \n"
        "&nbsp;&nbsp;\n"
        "<span style='background-color:#ff9999; color:black; padding: 2px 6px; border-radius: 4px; font-weight: bold;'>Keyword</span>\n"
        "<hr style='margin-top: 10px; margin-bottom: 10px;'>", 
        unsafe_allow_html=True
    )
    highlighted = highlight_text(doc_text, keywords, summary_sentences)
    st.markdown(
        f"""
        <div style="
            max-height: 500px;
            overflow-y: auto;
            padding: 1rem;
            border-radius: 8px;
            background-color: #111;
            line-height: 1.6;
            font-size: 16px;
        ">
        {highlighted}
        </div>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(layout="wide", page_title="NLP Analyzer")
st.title("NLP Research Analyzer")

st.sidebar.title("Configuration")
st.sidebar.info("Upload standard text or PDF documents, or use the pre-built research corpus to see traditional NLP techniques in action.")

preserve_numbers = st.sidebar.toggle(
    "Retain Numerical Data (e.g., statistics, years)",
    value=True
)

use_sample = st.sidebar.checkbox("Use Research Corpus", value=True)

uploaded_files = st.sidebar.file_uploader(
    "Upload text files",
    accept_multiple_files=True,
    disabled=use_sample
)

raw_docs = []
filenames = []

if use_sample:
    sample_dir = "research_documents"
    if os.path.exists(sample_dir):
        for filename in sorted(os.listdir(sample_dir)):
            if filename.endswith(".txt") or filename.endswith(".pdf"):
                path = os.path.join(sample_dir, filename)
                if filename.lower().endswith(".pdf"):
                    pdf = PdfReader(path)
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    raw_docs.append(text)
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        raw_docs.append(f.read())
                filenames.append(filename)
    else:
        st.error(f"Sample corpus directory '{sample_dir}' not found.")
elif uploaded_files:
    for file in uploaded_files:
        if file.name.lower().endswith(".pdf"):
            pdf = PdfReader(file)
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        else:
            text = file.getvalue().decode("utf-8", errors="ignore")
            
        raw_docs.append(text)
        filenames.append(file.name)

@st.cache_data
def run_pipeline(raw_docs, preserve_numbers):
    processed_docs = [
        execute_preprocessing_pipeline(doc, preserve_numeric=preserve_numbers) 
        for doc in raw_docs
    ]
    X, vectorizer = extract_tfidf_features(processed_docs)
    return processed_docs, X, vectorizer

if raw_docs:
    processed_docs, X, vectorizer = run_pipeline(raw_docs, preserve_numbers)

    # UI Enhancement: Dataset Quick Stats
    vocab_size = len(vectorizer.get_feature_names_out())
    
    st.markdown("### Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Documents", len(raw_docs))
    col2.metric("Vocabulary Size", f"{vocab_size} Terms")
    col3.metric("Clustering Engine", "K-Means++")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Clustering Insights", "LDA Topic Modeling", "Similarity Matrix"])

    with tab3:
        st.subheader("Document Similarity (Cosine Matrix)")
        if len(raw_docs) >= 2:
            similarity = calculate_cosine_similarity(X)
            sim_df = pd.DataFrame(similarity, index=filenames, columns=filenames)
            
            fig_sim = px.imshow(
                sim_df, 
                text_auto=".2f", 
                aspect="auto", 
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.warning("Please upload at least 2 documents to view similarity.")

    with tab2:
        st.subheader("Topic Extraction via Latent Dirichlet Allocation")
        if len(raw_docs) >= 2:
            st.write("Extracting underlying themes using Latent Dirichlet Allocation (LDA)...")
            n_topics = st.slider("Number of Topics", min_value=2, max_value=min(6, len(raw_docs)), value=3, key="lda_slider")
            lda_model = perform_lda_modeling(X, n_topics=n_topics)
            
            feature_names = vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(lda_model.components_):
                top_features_ind = topic.argsort()[:-10 - 1:-1]
                top_features = [feature_names[i] for i in top_features_ind]
                styled_theme = f"<span style='background-color:#0e1117; color:#50C878; padding: 4px 8px; border-radius: 4px; border: 1px solid #50C878; font-size: 16px;'>{', '.join(top_features)}</span>"
                st.markdown(f"**Theme {topic_idx + 1}:** &nbsp; {styled_theme}", unsafe_allow_html=True)
            st.divider()
        else:
            st.warning("Please upload at least 2 documents for topic modeling.")

    with tab1:
        st.subheader("Semantic K-Means Clustering")
        if len(raw_docs) >= 2:
            suggested_k = calculate_optimal_clusters(X)
            
            k = st.slider(
                "Number of Clusters (K-Means)",
                min_value=1,
                max_value=min(6, len(raw_docs)),
                value=suggested_k
            )
            
            if k > 0:
                labels = perform_kmeans_clustering(X, k)
                cluster_df = pd.DataFrame({"Document": filenames, "Cluster": labels})
                
                if k > 1:
                    coords = apply_dimensionality_reduction(X, n_components=2)
                    cluster_df['PCA1'] = coords[:, 0]
                    cluster_df['PCA2'] = coords[:, 1]
                    cluster_df['Cluster'] = cluster_df['Cluster'].astype(str)
                    
                    fig = px.scatter(
                        cluster_df, x='PCA1', y='PCA2', color='Cluster', 
                        hover_name='Document', title="Semantic Clustering (2D PCA)"
                    )
                    fig.update_traces(marker=dict(size=20, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
                    st.plotly_chart(fig)
                else:
                    st.info("K=1 Clustering Active. All documents are assigned to a single global cluster. 2D PCA graph is disabled since no separation analysis is required.")
                
                # Compute cluster-level features to show under cluster heading
                cluster_texts = {i: "" for i in range(k)}
                for label, text in zip(labels, raw_docs):
                    cluster_texts[label] += text + " "
                    
                cluster_list = [cluster_texts[i] for i in range(k)]
                
                processed_clusters = [execute_preprocessing_pipeline(c, preserve_numeric=preserve_numbers) for c in cluster_list]
                cluster_X, cluster_vectorizer = extract_tfidf_features(processed_clusters)
                
                cluster_vocab_size = len(cluster_vectorizer.get_feature_names_out())
                dynamic_top_n = max(
                    3,                        
                    min(
                        10,                   
                        int(0.1 * cluster_vocab_size)  
                    )
                )

                cluster_keywords = identify_top_keywords(
                    cluster_vectorizer, 
                    cluster_X, 
                    top_n=dynamic_top_n
                )
                
                cluster_summaries = [generate_extractive_summary(c, cluster_vectorizer, top_n=3) for c in cluster_list]

                st.subheader("📂 Cluster Insights")
                for cluster_id in range(k):
                    with st.container(border=True):
                        st.write(f"### Cluster {cluster_id}")
                        
                        # Show cluster-level insights
                        styled_keywords = f"<span style='background-color:#262730; color:#ffddb0; padding: 4px 8px; border-radius: 4px; font-size: 16px;'>{', '.join(cluster_keywords[cluster_id])}</span>"
                        st.markdown(f"**Keywords:**<br><br>{styled_keywords}", unsafe_allow_html=True)
                        st.markdown(f"<br>**Summary:**\n> {' '.join(cluster_summaries[cluster_id])}", unsafe_allow_html=True)
                        
                        cluster_docs_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                        st.write("**Documents:**")
                        for idx in cluster_docs_indices:
                            doc_name = filenames[idx]
                            if st.button(doc_name, key=f"btn_cluster_{cluster_id}_{idx}"):
                                show_document_modal(
                                    doc_name, 
                                    raw_docs[idx], 
                                    cluster_keywords[cluster_id], 
                                    cluster_summaries[cluster_id]
                                )
            else:
                
                st.subheader("📂 All Documents")
                global_keywords = identify_top_keywords(vectorizer, X, top_n=8)
                global_summaries = [generate_extractive_summary(doc, vectorizer, top_n=4) for doc in raw_docs]
                for idx, doc_name in enumerate(filenames):
                    if st.button(doc_name, key=f"btn_all_{idx}"):
                        show_document_modal(doc_name, raw_docs[idx], global_keywords[idx], global_summaries[idx])
        else:
            st.warning("Please upload at least 2 documents to view clustering.")
            
            st.subheader("📂 All Documents")
            global_keywords = identify_top_keywords(vectorizer, X, top_n=8)
            global_summaries = [generate_extractive_summary(doc, vectorizer, top_n=4) for doc in raw_docs]
            for idx, doc_name in enumerate(filenames):
                if st.button(doc_name, key=f"btn_single_{idx}"):
                    show_document_modal(doc_name, raw_docs[idx], global_keywords[idx], global_summaries[idx])