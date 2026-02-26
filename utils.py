import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import nltk

def generate_extractive_summary(text, vectorizer, top_n=3):
    # Use robust sentence tokenization instead of simple period split
    raw_sentences = nltk.sent_tokenize(text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]
    
    if not sentences:
        return []
        
    sentence_vectors = vectorizer.transform(sentences)
    sim_matrix = cosine_similarity(sentence_vectors)
    
    # Implement pure TextRank (PageRank applied to cosine similarity graph)
    nx_graph = nx.from_numpy_array(sim_matrix)
    
    try:
        scores = nx.pagerank(nx_graph)
    except Exception:
        # Fallback to pure degrees if graph convergence fails
        scores = {i: sim_matrix.sum(axis=1)[i] for i in range(len(sentences))}

    # Make sure we don't try to extract more sentences than exist
    n_sentences = min(top_n, len(sentences))
    
    # Get indices of top ranked sentences based on PageRank score
    ranked_scored = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
    top_indices = [idx for score, idx in ranked_scored[:n_sentences]]
    
    # Sort indices so the summary reads chronologically
    chronological_indices = sorted(top_indices)
    
    summary = []
    for i in chronological_indices:
        summary.append(sentences[i])

    return summary