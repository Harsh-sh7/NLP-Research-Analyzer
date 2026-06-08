import asyncio
import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from pymongo.database import Database

from backend.app.db.session import get_db, get_client
from backend.app.core.config import settings
from backend.app.schemas.research import ResearchCreate, ResearchJobOut, ReportOut
from backend.app.api.deps import get_current_user
from backend.app.services.agent_service import AgentService, agent_pubsub
from backend.app.core.security import jwt

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _job_out(doc: dict) -> dict:
    return {
        "id": doc["_id"],
        "user_id": doc["user_id"],
        "query": doc["query"],
        "status": doc["status"],
        "task_list": doc.get("task_list", []),
        "scraped_urls": doc.get("scraped_urls", []),
        "citations": doc.get("citations", {}),
        "report_draft": doc.get("report_draft", ""),
        "revision_count": doc.get("revision_count", 0),
        "created_at": doc["created_at"],
    }


def _report_out(doc: dict) -> dict:
    return {
        "id": doc["_id"],
        "job_id": doc["job_id"],
        "user_id": doc["user_id"],
        "title": doc["title"],
        "content": doc["content"],
        "metrics": doc["metrics"],
        "citations": doc["citations"],
        "created_at": doc["created_at"],
    }


def _get_ws_user(token: str) -> dict | None:
    """Verify JWT from a WebSocket query parameter and return the user doc."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            return None
        db = get_client()[settings.MONGODB_DB_NAME]
        return db["users"].find_one({"_id": user_id})
    except Exception:
        return None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/", response_model=ResearchJobOut)
async def create_research_job(
    payload: ResearchCreate,
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty",
        )

    job_doc = {
        "_id": str(uuid.uuid4()),
        "user_id": current_user["_id"],
        "query": payload.query.strip(),
        "status": "pending",
        "task_list": [],
        "scraped_urls": [],
        "citations": {},
        "report_draft": "",
        "revision_count": 0,
        "created_at": datetime.utcnow(),
    }
    db["research_jobs"].insert_one(job_doc)

    AgentService.start_research_task(job_doc["_id"], job_doc["query"], current_user["_id"])

    return _job_out(job_doc)


@router.get("/jobs", response_model=List[ResearchJobOut])
def list_research_jobs(
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    jobs = db["research_jobs"].find(
        {"user_id": current_user["_id"]}
    ).sort("created_at", -1)
    return [_job_out(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=ResearchJobOut)
def get_research_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    job = db["research_jobs"].find_one(
        {"_id": job_id, "user_id": current_user["_id"]}
    )
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Research run not found.",
        )
    return _job_out(job)


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_research_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    result = db["research_jobs"].delete_one(
        {"_id": job_id, "user_id": current_user["_id"]}
    )
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Research run not found.",
        )
    return None


@router.get("/reports", response_model=List[ReportOut])
def list_reports(
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    reports = db["reports"].find(
        {"user_id": current_user["_id"]}
    ).sort("created_at", -1)
    return [_report_out(r) for r in reports]


@router.get("/reports/{report_id}", response_model=ReportOut)
def get_report(
    report_id: str,
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    report = db["reports"].find_one(
        {"_id": report_id, "user_id": current_user["_id"]}
    )
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found.",
        )
    return _report_out(report)


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str, token: str):
    await websocket.accept()

    user = _get_ws_user(token)
    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    db = get_client()[settings.MONGODB_DB_NAME]
    job = db["research_jobs"].find_one(
        {"_id": job_id, "user_id": user["_id"]}
    )
    if not job:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    queue = agent_pubsub.subscribe(job_id)

    try:
        # Immediately replay final state if job already completed
        if job.get("status") == "completed":
            report = db["reports"].find_one({"job_id": job_id})
            if report:
                await websocket.send_json({
                    "type": "completed",
                    "report_draft": report["content"],
                    "metrics": report["metrics"],
                    "citations": report["citations"],
                })
        elif job.get("status") == "failed":
            await websocket.send_json({
                "type": "failed",
                "error": "This job previously failed.",
            })

        while True:
            data = await queue.get()
            await websocket.send_json(data)
            if data.get("type") in ["completed", "failed"]:
                break

    except WebSocketDisconnect:
        pass
    finally:
        agent_pubsub.unsubscribe(job_id, queue)
