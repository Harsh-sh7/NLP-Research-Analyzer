# 🧠 NLP Research Analyzer

This project is a traditional Natural Language Processing (NLP) system designed to analyze and summarize research documents locally, entirely without the need for large language models (LLMs) or external APIs.

## Features Included
- **Preprocessing:** Tokenization, Lemmatization, and Stop-word removal utilizing NLTK.
- **Feature Extraction:** TF-IDF vector corpus generation.
- **Topic Modeling:** Latent Dirichlet Allocation (LDA) applied to categorize topics based on keywords.
- **Clustering:** K-Means clustering algorithm visualized via PCA dimensionality reduction.
- **Extractive Summarization:** Custom Cosine-similarity sentence scoring mechanism for extractive summaries.
- **User Interface:** Streamlit web app with custom highlighting algorithms for qualitative review.

## Architecture
You can test this project directly by using the provided `create_corpus.py` data to populate research documents, then using `app.py` directly.

The structure is intentionally modular and uses basic machine learning algorithms ensuring zero API costs or privacy risks.
