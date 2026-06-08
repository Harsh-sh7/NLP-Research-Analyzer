from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class ResearchCreate(BaseModel):
    query: str


class ReportOut(BaseModel):
    id: str
    job_id: str
    title: str
    content: str
    metrics: Dict[str, Any]
    citations: Dict[str, Any]
    created_at: datetime


class ResearchJobOut(BaseModel):
    id: str
    query: str
    status: str
    task_list: List[Any]
    scraped_urls: List[str]
    citations: Dict[str, Any]
    report_draft: str
    revision_count: int
    created_at: datetime
    report: Optional[ReportOut] = None
