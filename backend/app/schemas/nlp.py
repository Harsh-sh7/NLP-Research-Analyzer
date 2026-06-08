from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class AnalysisParams(BaseModel):
    vectorization_mode: str = "TF-IDF (Classical)"  # or "Semantic Embeddings (SBERT)"
    k_clusters: int = 3
    preserve_numbers: bool = True
    n_topics: int = 3

class AnalysisRun(BaseModel):
    document_ids: List[str]
    parameters: Optional[AnalysisParams] = None

class AnalysisJobOut(BaseModel):
    id: str
    document_ids: List[str]
    vectorization_mode: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True
