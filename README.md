# NLP Research Analyzer

A Streamlit-powered Natural Language Processing application that analyses, compares, and clusters multiple documents (text or PDF) simultaneously. The tool offers dual vectorization modes — classical TF-IDF and Semantic Embeddings (SBERT + PCA) — enabling users to discover underlying themes and patterns across document collections with interactive visualizations and a premium dark-mode interface.

## Problem Statement

Given a heterogeneous collection of research documents, the goal is to:
1. Quantify pairwise similarity between documents (lexical or semantic)
2. Automatically group documents into coherent thematic clusters
3. Extract representative keywords and generate extractive summaries per cluster
4. Surface latent topics across the entire corpus
5. Provide interpretable, interactive visualizations for each analysis

The system supports two vectorization modes: a classical TF-IDF baseline for full explainability, and a semantic embedding mode (Sentence-BERT + PCA) for higher clustering accuracy.

## Features

- **Dual-Mode Vectorization:** Switch between TF-IDF (Classical) and Semantic Embeddings (SBERT) via a sidebar radio toggle.
- **Multi-Format Document Ingestion:** Upload multiple `.txt` or `.pdf` files, or choose from three built-in demo corpora (research text, AI research PDFs, semantic limitation demo).
- **Preprocessing Pipeline:** Tokenization, POS-aware lemmatization, and stopword removal. Includes a toggle to preserve numerical values, decimals, and percentage symbols within the text.
- **TF-IDF Vectorization:** Converts documents into numerical vectors using Term Frequency–Inverse Document Frequency with unigram + bigram support and dynamic vocabulary scaling.
- **Semantic Embeddings (SBERT + PCA):** Encodes documents using a pre-trained Sentence-BERT encoder (`all-MiniLM-L6-v2`) with chunk-averaged encoding, exponential-decay weighting, and PCA dimensionality reduction for high-accuracy clustering.
- **Cosine Similarity Heatmap:** Visualizes pairwise document similarity using an interactive Plotly heatmap with truncated axis labels and full-name hover tooltips.
- **K-Means Clustering & Silhouette Analysis:** Groups documents automatically based on content. Evaluates silhouette scores across multiple values of *k* on an interactive line chart, recommends the optimal cluster count, and allows manual slider override.
- **PCA Cluster Visualization:** Projects the high-dimensional vectors into 2D space using PCA (via Truncated SVD) for an interactive scatter plot of clusters.
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
   - **TF-IDF Mode:** Sublinear TF scaling, unigrams + bigrams (`ngram_range=(1, 2)`), dynamic `max_features` (60% of unique vocabulary, bounded 50–3000)
   - **Semantic Mode:** Sentence-BERT encoder (`all-MiniLM-L6-v2`) with 384-char chunk-averaged encoding, exponential-decay weighting, PCA reduction to 2 components, and L2 normalization

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
   - Quality guards: citation/reference detection, math-extraction noise filter, and cosine-based deduplication (70% threshold) to prevent near-duplicate sentences

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

### Results

| Corpus | Mode | Silhouette (k) | Cluster Accuracy |
|--------|------|:-:|:-:|
| Primary (9 text docs) | Semantic (SBERT + PCA) | **0.9902** (k=3) | 100% |
| Primary (9 text docs) | TF-IDF (Classical) | 0.1533 (k=3) | 100% |
| PDF Papers (7 docs) | Semantic (SBERT + PCA) | 0.2482 (k=2) | — |
| PDF Papers (7 docs) | TF-IDF (Classical) | 0.0713 (k=2) | — |

Semantic mode achieves a **6.5× improvement** over TF-IDF on the primary corpus.

## Optimization

- Implemented dynamic TF-IDF `max_features` scaling (60% of unique vocabulary, bounded 50–3000) to balance richness and sparsity.
- Applied sublinear term frequency (`sublinear_tf=True`) to dampen the effect of very high raw counts and improve discriminative power.
- Used `cosine` metric in silhouette scoring — more appropriate for L2-normalized vectors.
- Added K-Means++ initialisation (`init='k-means++'`, `n_init=10`) for more stable cluster convergence.
- Applied PCA dimensionality reduction on SBERT embeddings (384 → 2 components) to concentrate cluster signal and remove noise dimensions.
- Constrained extractive summarization input to 50,000 characters and filtered sentences to 20–500 characters to prevent table-of-contents junk and PDF artifacts from polluting summaries.
- Added citation/reference line detection (numbered entries, bibliography markers like "et al", "Proceedings", "doi:") to exclude non-content sentences from summaries.
- Implemented cosine-based near-duplicate removal (70% similarity threshold) so summaries contain diverse, non-repetitive sentences.
- Added math-extraction noise filter (single-char token ratio, real-word ratio, average token length) to handle formula-heavy PDFs.

## Assumptions & Reasoning

*   **Lexical Importance (TF-IDF):** The tool operates under the assumption that the relative frequency of specific terms and multi-word phrases across a corpus is a reliable proxy for a document's central themes. If a term appears frequently in one document but rarely in others, it is likely characteristic of that document's topic.
*   **Dimensionality and Distance (Cosine Similarity):** Cosine distance is chosen over Euclidean distance because it normalises against arbitrary document lengths. A 10-page and a 2-page paper on the same subject will still score highly, since only the direction (not magnitude) of their term vectors matters.
*   **Cluster Structure (K-Means):** It is assumed that thematically similar documents will naturally group together into relatively compact, separable regions in the high-dimensional TF-IDF vector space.
*   **Transparency (TextRank Extractive Summarization):** Sentences are selected directly from the original text based on their graph centrality scores, rather than being generated by an LLM. This approach eliminates the risk of hallucination and ensures that every sentence in the summary is traceable to a specific location in the source document.

## Limitations

*   **TF-IDF — Lack of Semantic Understanding:** TF-IDF relies entirely on exact string matching (lexical similarity). It cannot recognise synonyms, paraphrases, or contextual meaning. The Semantic Limitation Demo illustrates this explicitly.
*   **TF-IDF — Order Agnostic:** TF-IDF is a "Bag of Words" representation. While bigrams capture some local context, paragraph structure, grammar, and discourse flow are entirely ignored.
*   **Semantic Mode — Model Dependency:** The SBERT mode requires the `all-MiniLM-L6-v2` model (~90 MB download on first run) and PyTorch as a runtime dependency.
*   **Dimensionality & Sparsity:** Scaling to thousands of documents with very large vocabularies creates sparse matrices that may degrade clustering quality.

## Built-in Corpora

The application ships with three sample corpora to demonstrate its capabilities and constraints.

### 1. Primary Research Corpus
*   **Contents:** Nine text documents across three distinct domains — Quantum Computing, Cybersecurity, and Telemedicine (three documents per domain).
*   **Purpose:** Demonstrates standard multi-topic clustering. Documents within the same domain share heavy vocabulary overlap, allowing K-Means to group them correctly. Cross-domain similarity scores remain low, confirming effective separation.

### 2. Research Papers (PDF)
*   **Contents:** Seven PDF research papers spanning diverse domains — *A Model-Free Universal AI*, *EmpiRE-Compass*, *Generative Agents Navigating Digital Libraries*, *Iconographic Classification for Digitized Artworks*, *Learning-based Multi-agent Race Strategies in Formula 1*, *SPM-Bench*, and *Toward Expert Investment Teams*.
*   **Purpose:** Demonstrates cross-domain clustering on real academic papers. K-Means groups papers with overlapping vocabulary (e.g., AI/agent-focused papers vs. domain-specific ones) and the silhouette score helps identify natural topic boundaries.

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
