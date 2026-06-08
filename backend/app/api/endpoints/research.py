import asyncio
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from sqlalchemy.orm import Session
from typing import List

from backend.app.db.session import get_db, SessionLocal
from backend.app.db.models import ResearchJob, Report, User
from backend.app.schemas.research import ResearchCreate, ResearchJobOut, ReportOut
from backend.app.api.deps import get_current_user
from backend.app.services.agent_service import AgentService, agent_pubsub
from backend.app.core.security import jwt, settings

router = APIRouter()

# Helper to verify JWT from WebSocket query token
def get_ws_user(token: str, db: Session) -> User:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return db.query(User).filter(User.id == user_id).first()
    except Exception:
        return None

@router.post("/", response_model=ResearchJobOut)
async def create_research_job(
    payload: ResearchCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
        
    # Create ResearchJob entry
    job = ResearchJob(
        user_id=current_user.id,
        query=payload.query.strip(),
        status="pending"
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Start background execution
    AgentService.start_research_task(job.id, job.query, current_user.id)
    
    return job

@router.get("/jobs", response_model=List[ResearchJobOut])
def list_research_jobs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(ResearchJob).filter(
        ResearchJob.user_id == current_user.id
    ).order_by(ResearchJob.created_at.desc()).all()

@router.get("/jobs/{job_id}", response_model=ResearchJobOut)
def get_research_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    job = db.query(ResearchJob).filter(
        ResearchJob.id == job_id,
        ResearchJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Research run not found."
        )
    return job

@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_research_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    job = db.query(ResearchJob).filter(
        ResearchJob.id == job_id,
        ResearchJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Research run not found."
        )
        
    db.delete(job)
    db.commit()
    return None

@router.get("/reports", response_model=List[ReportOut])
def list_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(Report).filter(
        Report.user_id == current_user.id
    ).order_by(Report.created_at.desc()).all()

@router.get("/reports/{report_id}", response_model=ReportOut)
def get_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.user_id == current_user.id
    ).first()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found."
        )
    return report

@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str, token: str):
    await websocket.accept()
    
    db: Session = SessionLocal()
    user = get_ws_user(token, db)
    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        db.close()
        return
        
    # Confirm job belongs to user
    job = db.query(ResearchJob).filter(
        ResearchJob.id == job_id,
        ResearchJob.user_id == user.id
    ).first()
    
    if not job:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        db.close()
        return
        
    db.close()
    
    # Subscribe to the updates
    queue = agent_pubsub.subscribe(job_id)
    
    try:
        # If the job is already completed or failed, send the cached report/status immediately
        if job.status == "completed" and job.report:
            await websocket.send_json({
                "type": "completed",
                "report_draft": job.report.content,
                "metrics": job.report.metrics,
                "citations": job.report.citations
            })
        elif job.status == "failed":
            await websocket.send_json({
                "type": "failed",
                "error": "This job previously failed."
            })
            
        while True:
            # Wait for broadcasts from the background execution loop
            data = await queue.get()
            await websocket.send_json(data)
            
            # Stop listening if final state reached
            if data.get("type") in ["completed", "failed"]:
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        agent_pubsub.unsubscribe(job_id, queue)
