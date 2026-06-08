import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional

# Add workspace root to system path to reuse original logic
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.append(WORKSPACE_ROOT)

from backend import preprocessing
from backend import modeling
from backend import utils

class NLPService:
    @staticmethod
    def run_analysis(documents: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the complete NLP analysis pipeline on a list of document dicts:
        documents = [{"id": doc_id, "filename": name, "content": text}, ...]
        params = {"vectorization_mode": str, "k_clusters": int, "preserve_numbers": bool, "n_topics": int}
        """
        raw_docs = [doc["content"] for doc in documents]
        filenames = [doc["filename"] for doc in documents]
        doc_ids = [doc["id"] for doc in documents]
        
        if not raw_docs:
            return {"error": "No documents provided"}
            
        vectorization_mode = params.get("vectorization_mode", "TF-IDF (Classical)")
        use_semantic = "SBERT" in vectorization_mode
        preserve_numbers = params.get("preserve_numbers", True)
        k_clusters = params.get("k_clusters", 3)
        n_topics = params.get("n_topics", 3)
        
        # 1. Preprocess
        processed_docs = [
            preprocessing.execute_preprocessing_pipeline(doc, preserve_numeric=preserve_numbers)
            for doc in raw_docs
        ]
        
        # 2. Extract TF-IDF (always needed for keywords, summaries, LDA)
        X_tfidf, vectorizer = modeling.extract_tfidf_features(processed_docs)
        
        # 3. Choose Clustering Space (TF-IDF vs SBERT)
        if use_semantic:
            X = modeling.compute_semantic_embeddings(raw_docs)
        else:
            X = X_tfidf
            
        n_docs = len(raw_docs)
        
        # 4. Silhouette Analysis (for k=2..min(10, n_docs-1))
        suggested_k, scores_per_k = modeling.calculate_optimal_clusters(X)
        
        # 5. K-Means
        # Ensure k is valid
        k = max(1, min(k_clusters, n_docs))
        labels = modeling.perform_kmeans_clustering(X, k=k)
        
        # 6. Dimensionality Reduction (PCA for 2D projection)
        if n_docs > 1:
            if use_semantic:
                # SBERT is already reduced to 2D PCA inside compute_semantic_embeddings
                coords = X
            else:
                coords = modeling.apply_dimensionality_reduction(X, n_components=2)
            pca_data = [
                {
                    "id": doc_ids[i],
                    "name": filenames[i],
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]) if coords.shape[1] > 1 else 0.0,
                    "cluster": int(labels[i])
                }
                for i in range(n_docs)
            ]
        else:
            pca_data = [
                {
                    "id": doc_ids[0],
                    "name": filenames[0],
                    "x": 0.0,
                    "y": 0.0,
                    "cluster": 0
                }
            ]
            
        # 7. Cosine Similarity Heatmap
        similarity_matrix = modeling.calculate_cosine_similarity(X)
        similarity_data = {
            "matrix": similarity_matrix.tolist(),
            "filenames": filenames,
            "ids": doc_ids
        }
        
        # 8. LDA Topics
        topics = []
        if n_docs >= 2:
            lda_model = modeling.perform_lda_modeling(X_tfidf, n_topics=min(n_topics, n_docs))
            feature_names = vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(lda_model.components_):
                top_features_ind = topic.argsort()[:-10 - 1:-1]
                top_features = [feature_names[i] for i in top_features_ind]
                topics.append(", ".join(top_features))
                
        # 9. Extract Cluster Keywords & Extractive Summaries
        cluster_texts = {i: "" for i in range(k)}
        for label, text in zip(labels, raw_docs):
            cluster_texts[label] += text + " "
            
        cluster_list = [cluster_texts[i] for i in range(k)]
        processed_clusters = [
            preprocessing.execute_preprocessing_pipeline(c, preserve_numeric=preserve_numbers) 
            for c in cluster_list
        ]
        
        cluster_X, cluster_vectorizer = modeling.extract_tfidf_features(processed_clusters)
        cluster_vocab_size = len(cluster_vectorizer.get_feature_names_out())
        dynamic_top_n = max(3, min(10, int(0.1 * cluster_vocab_size)))
        cluster_keywords = modeling.identify_top_keywords(cluster_vectorizer, cluster_X, top_n=dynamic_top_n)
        
        # Format keywords as a list of lists of strings
        formatted_cluster_keywords = [[str(word) for word in kws] for kws in cluster_keywords]
        
        # Generate document-level summary inside each cluster and extract keywords per document
        doc_analysis = {}
        for idx, doc_id in enumerate(doc_ids):
            readable = preprocessing.prepare_text_for_summary(raw_docs[idx], preserve_numeric=preserve_numbers)
            doc_sents = utils.generate_extractive_summary(readable, cluster_vectorizer, top_n=2)
            
            # Keywords for this single document in original TF-IDF space
            sorted_indices = X_tfidf[idx].toarray().flatten().argsort()[-8:]
            all_feature_names = vectorizer.get_feature_names_out()
            doc_kws = [all_feature_names[i] for i in sorted_indices]
            
            doc_analysis[doc_id] = {
                "summary": doc_sents,
                "keywords": doc_kws,
                "cleaned_text": readable,
                "raw_text": raw_docs[idx]
            }
            
        # Group by cluster ID
        cluster_data = []
        for cid in range(k):
            doc_indices = [i for i, lbl in enumerate(labels) if lbl == cid]
            cluster_docs = []
            for idx in doc_indices:
                did = doc_ids[idx]
                cluster_docs.append({
                    "id": did,
                    "filename": filenames[idx],
                    "summary": doc_analysis[did]["summary"],
                    "keywords": doc_analysis[did]["keywords"]
                })
            cluster_data.append({
                "cluster_id": cid,
                "keywords": formatted_cluster_keywords[cid],
                "documents": cluster_docs
            })
            
        # Convert keys in scores_per_k to strings for MongoDB compatibility
        scores_per_k_str = {str(k_val): val for k_val, val in scores_per_k.items()}

        return {
            "vectorization_mode": vectorization_mode,
            "k_clusters": k,
            "suggested_k": suggested_k,
            "scores_per_k": scores_per_k_str,
            "pca_scatter": pca_data,
            "similarity": similarity_data,
            "topics": topics,
            "clusters": cluster_data,
            "document_details": doc_analysis  # contains highlighted detail data per document id
        }
