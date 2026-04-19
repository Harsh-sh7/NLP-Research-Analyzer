# Unified Research Platform

A Streamlit-powered Multi-Page Application that combines **Classical Natural Language Processing** algorithms with an **Agentic Generative AI** swarm to create an end-to-end ecosystem for document analysis and research synthesis.

The platform provides two primary modes of operation, accessible via the sidebar navigation:

1. **Classical NLP Analysis**: Analyze, compare, and cluster your own private research documents (text or PDF) using TF-IDF, Semantic Embeddings (SBERT), K-Means, and Latent Dirichlet Allocation.
2. **Agentic GenAI Research**: Deploy an autonomous agent to break down a prompt, search the web, scrape articles, store them in a local vector database, and compile fully cited, multi-node reports.

---

## 1. Classical NLP Analysis

Given a heterogeneous collection of research documents, this sub-system allows you to:
1. Quantify pairwise similarity between documents (lexical or semantic)
2. Automatically group documents into coherent thematic clusters
3. Extract representative keywords and generate extractive summaries per cluster
4. Surface latent topics across the entire corpus
5. Provide interpretable, interactive visualizations for each analysis

### NLP Features
- **Dual-Mode Vectorization:** Switch between TF-IDF (Classical) and Semantic Embeddings (SBERT).
- **Multi-Format Document Ingestion:** Upload multiple `.txt` or `.pdf` files.
- **Preprocessing Pipeline:** Tokenization, POS-aware lemmatization, and stopword removal.
- **TF-IDF Vectorization:** Sublinear TF scaling with dynamic max_features.
- **Semantic Embeddings (SBERT + PCA):** Encodes docs using Sentence-BERT (`all-MiniLM-L6-v2`) with chunk-averaged encoding and exponential decay weighting.
- **K-Means Clustering:** Groups documents automatically with Silhouette Score optimization.
- **LDA Topic Modeling:** Discovers hidden latent themes across the corpus.
- **Extractive Summarization (TextRank):** Generates traceable summaries for each cluster.

---

## 2. Agentic GenAI Research

An autonomous, multi-node research agent built with **LangGraph** that decomposes complex queries, searches the live web, scrapes and vectorizes pages with **RAG**, synthesizes structured Markdown reports, and self-reviews its own output. 

### GenAI Features
- **Query Decomposition** — Automatically breaks a broad research topic into 3–5 targeted sub-questions.
- **Live Web Search** — Searches the internet in real-time via the Tavily API.
- **Deep Web Scraping & RAG** — Visits top URLs, chunks text, embeds them via `FAISS` and performs retrieval-augmented generation. 
- **Structured Report Synthesis** — Generates a professional Markdown report with references.
- **Automated Quality Review** — A Reviewer node validates the draft against coverage criteria, triggering rework cycles if quality thresholds aren't met.

### GenAI Architecture
Built as a **LangGraph StateGraph** mapped out across 5 sequential nodes:
- **Planner**: Llama 3.1 8B — Decomposes query
- **Researcher**: Tavily — Harvests top search results
- **Scraper**: BS4 + FAISS — Vectorizes and retrieves relevant context snippets
- **Synthesizer**: Llama 3.3 70B — Stitches findings into cohesive markdown
- **Reviewer**: Llama 3.1 8B — Validates the result and triggers feedback loops

---

## Installation & Setup

1. **Clone the repository** (if you haven't already).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure API Keys (For Agentic Research Mode)**:
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and configure your API keys:
   - **GROQ_API_KEY**: For powering LLM generation / Node Agents ([Console](https://console.groq.com/))
   - **TAVILY_API_KEY**: For live web searching ([Tavily](https://app.tavily.com/))

## Run Locally

Launch the app using Streamlit:
```bash
streamlit run app.py
```

The application will open an elegant central dashboard. Use the **sidebar** to navigate between modes.

---

## Project Structure

```
├── app.py                            # Streamlit Homepage/Landing
├── pages/
│   ├── 1_Classical_NLP_Analysis.py   # Milestone 1 Code base
│   └── 2_Agentic_GenAI_Research.py   # Milestone 2 Code base
├── agent/                            # LangGraph workflow code
├── preprocessing.py                  # Text cleaning (NLP Mode)
├── modeling.py                       # ML models (NLP Mode)
├── utils.py                          # Summarization utils (NLP Mode)
├── create_corpus.py                  
├── requirements.txt
├── .env.example
└── research_documents/               # Demo corpus (NLP Mode)
```
