# NLP Research Analyzer

A Streamlit-powered Natural Language Processing application that analyses, compares, and clusters multiple documents (text or PDF) simultaneously. The tool enables users to discover underlying themes and patterns across document collections using classical NLP techniques, with full explainability, interactive visualizations, and a premium dark-mode interface.

## Problem Statement

Given a heterogeneous collection of research documents, the goal is to:
1. Quantify pairwise lexical similarity between documents
2. Automatically group documents into coherent thematic clusters
3. Extract representative keywords and generate extractive summaries per cluster
4. Surface latent topics across the entire corpus
5. Provide interpretable, interactive visualizations for each analysis

The challenge lies in achieving meaningful document understanding using only classical NLP methods — without relying on modern semantic embeddings or large language models.

## Features

- **Multi-Format Document Ingestion:** Upload multiple `.txt` or `.pdf` files, or choose from three built-in demo corpora (research text, AI research PDFs, semantic limitation demo).
- **Preprocessing Pipeline:** Tokenization, POS-aware lemmatization, and stopword removal. Includes a toggle to preserve numerical values, decimals, and percentage symbols within the text.
- **TF-IDF Vectorization:** Converts documents into numerical vectors using Term Frequency–Inverse Document Frequency with unigram + bigram support and dynamic vocabulary scaling.
- **Cosine Similarity Heatmap:** Visualizes pairwise document similarity using an interactive Plotly heatmap with truncated axis labels and full-name hover tooltips.
- **K-Means Clustering & Silhouette Analysis:** Groups documents automatically based on content. Evaluates silhouette scores across multiple values of *k* on an interactive line chart, recommends the optimal cluster count, and allows manual slider override.
- **PCA Cluster Visualization:** Projects the high-dimensional TF-IDF vectors into 2D space using PCA (via Truncated SVD) for an interactive scatter plot of clusters.
- **LDA Topic Modeling:** Discovers hidden latent themes across the corpus using Latent Dirichlet Allocation, displaying top terms per topic.
- **Cluster Introspection:** For each cluster, automatically extracts characteristic **keywords** (displayed as styled pills) and generates a multi-sentence **extractive summary** using a TextRank algorithm (PageRank over a sentence similarity graph).
- **Document Modal Viewer:** Click individual documents within a cluster to open a two-tab popup dialog — one tab showing the original unprocessed text, and another with keywords and summary sentences interactively highlighted.
- **Educational Expanders:** Collapsible information panels under each major analysis section explaining what the NLP technique does and why it matters.

## Methodology

1. Lexical Preprocessing
   - Tokenization (NLTK word tokenizer)
   - POS-aware Lemmatization (WordNet + Treebank POS mapping)
   - Stopword Removal (NLTK English stopwords)
   - Optional Numeric Preservation (retains values like `5.2%`, `2023`)

2. Vector Representation
   - TF-IDF with sublinear term frequency scaling
   - Unigrams + Bigrams (`ngram_range=(1, 2)`)
   - Dynamic `max_features` scaling (50% of unique vocabulary, bounded between 50–1000)

3. Similarity Computation
   - Cosine Similarity (normalized dot product, length-agnostic)

4. Clustering
   - K-Means++ initialisation with 10 restarts
   - Silhouette Score (cosine metric) for automatic *k* selection
   - 2D PCA projection via Truncated SVD for scatter visualization

5. Topic Modeling
   - Latent Dirichlet Allocation (LDA) with batch learning
   - Configurable number of topics (2–6) via interactive slider

6. Extractive Summarization
   - TextRank algorithm: builds a sentence similarity graph using cosine similarity of TF-IDF vectors, then applies PageRank to rank sentences by global importance
   - Sentence count scales dynamically with cluster size (~2 sentences per document, min 4, max 10)

7. Visualization & Interaction
   - Cosine Similarity Heatmap (Plotly, with truncated labels and full-name hover)
   - PCA Cluster Scatter Plot (Plotly)
   - Silhouette Score Line Chart (Plotly)
   - Interactive Document Modal with highlighted keywords and summary sentences

## Evaluation

Since this is an unsupervised system, traditional accuracy metrics (precision, recall, F1) do not apply.

Performance is evaluated using:
- **Silhouette Score** — measures cluster separation quality (range: −1 to 1; higher is better)
- **Intra-domain vs Inter-domain similarity margins** — documents from the same domain should score significantly higher in cosine similarity than cross-domain pairs
- **Qualitative keyword interpretability** — extracted keywords should be recognisable as domain-specific terms

## Optimization

- Implemented dynamic TF-IDF `max_features` scaling (proportional to unique vocabulary size) to control sparsity without manual tuning.
- Applied sublinear term frequency (`sublinear_tf=True`) to dampen the effect of very high raw counts and improve discriminative power.
- Used `cosine` metric in silhouette scoring instead of Euclidean — more appropriate for the L2-normalized TF-IDF vectors.
- Added K-Means++ initialisation (`init='k-means++'`, `n_init=10`) for more stable cluster convergence.
- Constrained extractive summarization input to 50,000 characters and filtered sentences to 20–500 characters to prevent table-of-contents junk and PDF artifacts from polluting summaries.

## Assumptions & Reasoning

*   **Lexical Importance (TF-IDF):** The tool operates under the assumption that the relative frequency of specific terms and multi-word phrases across a corpus is a reliable proxy for a document's central themes. If a term appears frequently in one document but rarely in others, it is likely characteristic of that document's topic.
*   **Dimensionality and Distance (Cosine Similarity):** Cosine distance is chosen over Euclidean distance because it normalises against arbitrary document lengths. A 10-page and a 2-page paper on the same subject will still score highly, since only the direction (not magnitude) of their term vectors matters.
*   **Cluster Structure (K-Means):** It is assumed that thematically similar documents will naturally group together into relatively compact, separable regions in the high-dimensional TF-IDF vector space.
*   **Transparency (TextRank Extractive Summarization):** Sentences are selected directly from the original text based on their graph centrality scores, rather than being generated by an LLM. This approach eliminates the risk of hallucination and ensures that every sentence in the summary is traceable to a specific location in the source document.

## Limitations

*   **Lack of Semantic Understanding:** TF-IDF relies entirely on exact string matching (lexical similarity). It cannot recognise synonyms, paraphrases, or contextual meaning. For example, "automobile" and "car" are treated as completely independent dimensions in the vector space, despite being semantically identical.
*   **Order Agnostic:** At its core, TF-IDF is a "Bag of Words" representation. While bigrams capture some local word co-occurrence context, overall paragraph structure, grammar, and discourse flow are entirely ignored.
*   **Dimensionality & Sparsity:** Scaling to thousands of documents with very large unique vocabularies creates highly sparse matrices. This can degrade clustering quality and may require aggressive dimensionality reduction or feature selection techniques.

## Built-in Corpora

The application ships with three sample corpora to demonstrate its capabilities and constraints.

### 1. Primary Research Corpus
*   **Contents:** Nine text documents across three distinct domains — Quantum Computing, Cybersecurity, and Telemedicine (three documents per domain).
*   **Purpose:** Demonstrates standard multi-topic clustering. Documents within the same domain share heavy vocabulary overlap, allowing K-Means to group them correctly. Cross-domain similarity scores remain low, confirming effective separation.

### 2. AI Research Papers (PDF)
*   **Contents:** Seven PDF files including landmark computer science research papers (*Attention Is All You Need*, *BERT*, *Deep Residual Learning*, *ImageNet Classification with Deep CNNs*, *MapReduce*, *The Google File System*) and one completely unrelated outlier (*Cricket Rule Book*).
*   **Purpose:** Showcases when this classical framework excels. The vocabulary domains are distinct and highly specialised (e.g., "attention", "transformer", "residual", "MapReduce", "innings", "umpire"). K-Means easily separates the papers into correct technical groupings and isolates the unrelated outlier.

### 3. Semantic Limitation Demo
*   **Contents:** Three very short text documents, each describing the same real-world event — an online shopping transaction — but each using entirely different vocabulary (e.g., "shopper placed an order" vs. "client procured a gadget" vs. "end-user acquired a device").
*   **Purpose:** Demonstrates the critical failure point of TF-IDF. Because these documents share virtually no overlapping words despite meaning the same thing, cosine similarity reports near-zero scores. This proves that classical lexical approaches cannot inherently link synonymous terms without relying on modern semantic embeddings.

## Project Structure

```
├── app.py                  # Streamlit application (UI + orchestration)
├── preprocessing.py        # Text cleaning, tokenization, lemmatization
├── modeling.py             # TF-IDF, K-Means, LDA, PCA, similarity
├── utils.py                # Summarization, visualizations, file I/O
├── create_corpus.py        # Script to regenerate the text research corpus
├── requirements.txt
├── research_documents/
│   ├── *.txt               # Primary research corpus (9 text files)
│   ├── pdf_papers/         # AI research paper PDFs (7 files)
│   └── semantic_demo/      # Semantic limitation demo texts (3 files)
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```
