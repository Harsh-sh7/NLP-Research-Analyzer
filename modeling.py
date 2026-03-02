import os
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), ".hf_cache"))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import numpy as np

# ---------------------------------------------------------------------------
# Semantic Embeddings (sentence-transformers — encoder only, no generative AI)
# ---------------------------------------------------------------------------

_sbert_model = None

def _get_sbert_model():
    """Lazy-load the sentence-transformer model (loads once, reuses)."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


def compute_semantic_embeddings(raw_docs, chunk_size=384):
    """Encode documents into dense semantic vectors via chunk-averaged SBERT + PCA."""
    from sklearn.decomposition import PCA
    model = _get_sbert_model()
    embeddings = []
    for text in raw_docs:
        chunks = [text[i:i+chunk_size] for i in range(0, min(len(text), 10_000), chunk_size)]
        if not chunks:
            chunks = [text[:chunk_size]]
        chunk_embs = model.encode(chunks, show_progress_bar=False)
        # Exponential-decay weighting: earlier chunks (abstract, intro) matter more
        weights = np.exp(-np.arange(len(chunk_embs)) * 0.15)
        avg = np.average(chunk_embs, axis=0, weights=weights)
        avg = avg / np.linalg.norm(avg)
        embeddings.append(avg)
    raw_matrix = np.array(embeddings)

    # PCA reduction — concentrates cluster signal, removes noise dimensions
    n_components = min(raw_matrix.shape[0] - 1, 2)
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(raw_matrix)
    reduced = normalize(reduced)
    return reduced

def dynamic_max_features(docs):
    """Calculate optimal TF-IDF vocabulary size as 60% of unique tokens, bounded [50, 3000]."""
    all_tokens = " ".join(docs).split()
    unique_vocab = len(set(all_tokens))
    
    # Keep 60% of unique vocabulary for richer representation
    max_features = int(0.6 * unique_vocab)
    
    max_features = max(50, max_features)    # lower bound
    max_features = min(3000, max_features)  # upper bound

    return max_features



def extract_tfidf_features(docs):
    """Build a TF-IDF matrix with dynamic vocabulary, bigrams, and sublinear TF scaling."""
    n_docs = len(docs)

    if n_docs < 3:
        min_df = 1
        max_df = 1.0
    else:
        min_df = 1
        max_df = 0.85  # remove overly common words
    max_features = dynamic_max_features(docs)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df= min_df,
        max_df = max_df,
        max_features= max_features,
        sublinear_tf=True,
        use_idf=True,
        norm='l2'
    )
    X = vectorizer.fit_transform(docs)

    return X, vectorizer

def calculate_cosine_similarity(X):
    """Compute pairwise cosine similarity matrix for the given feature matrix."""
    return cosine_similarity(X)

def perform_kmeans_clustering(X, k=3):
    """Run K-Means++ clustering and return cluster labels for each document."""
    if k == 1:
        return np.zeros(X.shape[0], dtype=int)
    model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = model.fit_predict(X)
    return labels

def calculate_optimal_clusters(X, max_k=10):
    """Evaluate silhouette scores for k=2..max_k and return (best_k, scores_dict)."""
    n_docs = X.shape[0]

    # We need at least 2 clusters and at most n_docs - 1
    upper_bound = min(max_k, n_docs - 1)

    best_k = 2
    best_score = -1
    scores_per_k = {}

    if n_docs < 3:
        # Silhouette requires at least 3 samples for meaningful range
        return best_k, scores_per_k

    for k in range(2, upper_bound + 1):
        try:
            model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            labels = model.fit_predict(X)

            # Guard against degenerate single-label clustering
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels, metric='cosine')
            else:
                score = -1

            scores_per_k[k] = round(float(score), 4)

            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            scores_per_k[k] = -1.0

    return best_k, scores_per_k

def identify_top_keywords(vectorizer, X, top_n=10):
    """Extract the top-N highest TF-IDF terms for each document vector."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    keywords = []

    for doc_vector in X:
        sorted_indices = doc_vector.toarray().flatten().argsort()[-top_n:]
        keywords.append(feature_names[sorted_indices])

    return keywords

def perform_lda_modeling(X, n_topics=3):
    """Fit a Latent Dirichlet Allocation model and return it for topic extraction."""
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20, learning_method='batch')
    lda.fit(X)
    return lda

def apply_dimensionality_reduction(X, n_components=2):
    """Reduce feature matrix to n_components dimensions using Truncated SVD."""
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(X)