import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from backend.app.db.session import get_db
from backend.app.schemas.nlp import AnalysisRun, AnalysisJobOut
from backend.app.api.deps import get_current_user
from backend.app.services.nlp_service import NLPService

router = APIRouter()


def _job_out(doc: dict) -> dict:
    return {
        "id": doc["_id"],
        "user_id": doc["user_id"],
        "document_ids": doc["document_ids"],
        "vectorization_mode": doc["vectorization_mode"],
        "parameters": doc["parameters"],
        "results": doc["results"],
        "created_at": doc["created_at"],
    }


@router.post("/analyze", response_model=AnalysisJobOut)
def run_nlp_analysis(
    payload: AnalysisRun,
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    user_id = current_user["_id"]

    # Fetch requested documents owned by this user
    docs = list(
        db["documents"].find(
            {"_id": {"$in": payload.document_ids}, "user_id": user_id}
        )
    )

    if len(docs) < len(payload.document_ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="One or more specified documents could not be found or do not belong to you.",
        )

    if not docs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents selected for analysis.",
        )

    doc_dicts = [
        {"id": d["_id"], "filename": d["filename"], "content": d["content"]}
        for d in docs
    ]

    params = payload.parameters.dict() if payload.parameters else {
        "vectorization_mode": "TF-IDF (Classical)",
        "k_clusters": 3,
        "preserve_numbers": True,
        "n_topics": 3,
    }

    try:
        results = NLPService.run_analysis(doc_dicts, params)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NLP pipeline execution failed: {str(e)}",
        )

    job_doc = {
        "_id": str(uuid.uuid4()),
        "user_id": user_id,
        "document_ids": payload.document_ids,
        "vectorization_mode": params.get("vectorization_mode", "TF-IDF (Classical)"),
        "parameters": params,
        "results": results,
        "created_at": datetime.utcnow(),
    }
    db["analysis_jobs"].insert_one(job_doc)

    return _job_out(job_doc)


@router.get("/jobs", response_model=List[AnalysisJobOut])
def list_analysis_jobs(
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    jobs = db["analysis_jobs"].find(
        {"user_id": current_user["_id"]}
    ).sort("created_at", -1)
    return [_job_out(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=AnalysisJobOut)
def get_analysis_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    job = db["analysis_jobs"].find_one(
        {"_id": job_id, "user_id": current_user["_id"]}
    )
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="NLP Analysis run not found.",
        )
    return _job_out(job)
