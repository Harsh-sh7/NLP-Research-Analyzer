import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def scrape_url(url: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
        return "\n".join(lines)
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks


def retrieve_relevant_chunks(texts: list[str], query: str, k: int = 3) -> list[str]:
    if not texts:
        return []
    
    # Vectorize texts and query using scikit-learn (extremely fast and memory efficient)
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        query_vec = vectorizer.transform([query])
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Sort and select top k
        top_indices = np.argsort(similarities)[::-1][:k]
        return [texts[i] for i in top_indices]
    except Exception:
        # Fallback to returning the first k chunks if vectorization fails
        return texts[:k]

