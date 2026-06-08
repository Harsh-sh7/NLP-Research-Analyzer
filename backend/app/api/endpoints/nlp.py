from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from backend.app.db.session import get_db
from backend.app.db.models import Document, AnalysisJob, User
from backend.app.schemas.nlp import AnalysisRun, AnalysisJobOut
from backend.app.api.deps import get_current_user
from backend.app.services.nlp_service import NLPService

router = APIRouter()

@router.post("/analyze", response_model=AnalysisJobOut)
def run_nlp_analysis(
    payload: AnalysisRun,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Fetch documents from DB
    docs = db.query(Document).filter(
        Document.id.in_(payload.document_ids),
        Document.user_id == current_user.id
    ).all()
    
    if len(docs) < len(payload.document_ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="One or more specified documents could not be found or do not belong to you."
        )
        
    if not docs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents selected for analysis."
        )
        
    doc_dicts = [
        {"id": doc.id, "filename": doc.filename, "content": doc.content}
        for doc in docs
    ]
    
    # Pre-configure parameters
    params = {}
    if payload.parameters:
        params = payload.parameters.dict()
    else:
        params = {
            "vectorization_mode": "TF-IDF (Classical)",
            "k_clusters": 3,
            "preserve_numbers": True,
            "n_topics": 3
        }
        
    # Execute analysis
    try:
        results = NLPService.run_analysis(doc_dicts, params)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NLP pipeline execution failed: {str(e)}"
        )
        
    # Save to DB
    job = AnalysisJob(
        user_id=current_user.id,
        document_ids=payload.document_ids,
        vectorization_mode=params.get("vectorization_mode", "TF-IDF (Classical)"),
        parameters=params,
        results=results
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    return job

@router.get("/jobs", response_model=List[AnalysisJobOut])
def list_analysis_jobs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(AnalysisJob).filter(
        AnalysisJob.user_id == current_user.id
    ).order_by(AnalysisJob.created_at.desc()).all()

@router.get("/jobs/{job_id}", response_model=AnalysisJobOut)
def get_analysis_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    job = db.query(AnalysisJob).filter(
        AnalysisJob.id == job_id,
        AnalysisJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="NLP Analysis run not found."
        )
    return job
