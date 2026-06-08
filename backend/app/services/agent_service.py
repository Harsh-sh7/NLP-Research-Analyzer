import os
import sys
import asyncio
from datetime import datetime
from typing import Dict, List, Any
from sqlalchemy.orm import Session

# Add workspace root to system path to reuse original logic
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.append(WORKSPACE_ROOT)

from agent.graph import build_graph
from backend.app.db.session import SessionLocal
from backend.app.db.models import ResearchJob, Report

class AgentPubSub:
    def __init__(self):
        self._listeners: Dict[str, List[asyncio.Queue]] = {}

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q = asyncio.Queue()
        if job_id not in self._listeners:
            self._listeners[job_id] = []
        self._listeners[job_id].append(q)
        return q

    def unsubscribe(self, job_id: str, queue: asyncio.Queue):
        if job_id in self._listeners:
            try:
                self._listeners[job_id].remove(queue)
            except ValueError:
                pass
            if not self._listeners[job_id]:
                del self._listeners[job_id]

    def broadcast(self, job_id: str, data: Dict[str, Any]):
        if job_id in self._listeners:
            for q in self._listeners[job_id]:
                q.put_nowait(data)

agent_pubsub = AgentPubSub()

class AgentService:
    @staticmethod
    def start_research_task(job_id: str, query: str, user_id: str):
        """Starts the research agent run in a background asyncio Task."""
        asyncio.create_task(AgentService._run_agent_flow(job_id, query, user_id))

    @staticmethod
    async def _run_agent_flow(job_id: str, query: str, user_id: str):
        """Asynchronously runs the LangGraph and handles all state transitions, broadcasts, and database saves."""
        # Initialize db session inside background task
        db: Session = SessionLocal()
        
        try:
            job = db.query(ResearchJob).filter(ResearchJob.id == job_id).first()
            if not job:
                return
            
            job.status = "running"
            db.commit()
            
            agent = build_graph()
            
            initial_state = {
                "research_query": query,
                "task_list": [],
                "completed_tasks": [],
                "search_results": [],
                "scraped_urls": [],
                "context_library": [],
                "citations": {},
                "report_draft": "",
                "revision_count": 0,
                "reviewer_feedback": "",
                "status": "starting",
            }
            
            agent_pubsub.broadcast(job_id, {"type": "info", "message": "Research job initialized."})
            
            # Start streaming the graph nodes
            current_node = "planner"
            agent_pubsub.broadcast(job_id, {
                "type": "node_start",
                "node": current_node,
                "message": "Planner node active. Decomposing research query..."
            })
            
            # Since LangGraph stream is synchronous, run it in a separate thread so it doesn't block the async event loop
            loop = asyncio.get_running_loop()
            
            def run_graph_sync():
                events = []
                for event in agent.stream(initial_state):
                    events.append(event)
                return events
                
            # We want to handle events one by one as they happen. So let's run a generator-friendly thread loop
            def get_stream_iterator():
                return agent.stream(initial_state)
                
            iterator = await loop.run_in_executor(None, get_stream_iterator)
            
            # Custom wrapper to execute next() in a thread
            def get_next_event(it):
                try:
                    return next(it)
                except StopIteration:
                    return None
            
            final_state = {}
            iteration = 0
            
            while True:
                event = await loop.run_in_executor(None, get_next_event, iterator)
                if event is None:
                    break
                    
                for node_name, node_output in event.items():
                    if node_name.startswith("__"):
                        continue
                        
                    # Save intermediate states to final_state
                    if isinstance(node_output, dict):
                        final_state.update(node_output)
                        
                    # Broadcast node completion
                    agent_pubsub.broadcast(job_id, {
                        "type": "node_complete",
                        "node": node_name,
                        "data": {
                            "task_list": final_state.get("task_list", []),
                            "scraped_urls": final_state.get("scraped_urls", []),
                            "search_results_count": len(final_state.get("search_results", [])),
                            "revision_count": final_state.get("revision_count", 0),
                            "status": final_state.get("status", "")
                        }
                    })
                    
                    # Determine next node to highlight
                    next_node = None
                    if node_name == "planner":
                        next_node = "researcher"
                        msg = "Researcher node active. Searching web..."
                    elif node_name == "researcher":
                        next_node = "scraper"
                        msg = f"Scraper node active. Indexing {len(final_state.get('search_results', []))} results..."
                    elif node_name == "scraper":
                        next_node = "synthesizer"
                        msg = "Synthesizer node active. Drafting markdown report..."
                    elif node_name == "synthesizer":
                        next_node = "reviewer"
                        msg = "Reviewer node active. Assessing quality parameters..."
                    elif node_name == "reviewer":
                        if final_state.get("status") == "needs_revision" and final_state.get("revision_count", 0) < 2:
                            iteration += 1
                            next_node = "researcher"
                            msg = f"Revision cycle {iteration} triggered. Querying search engines again..."
                            agent_pubsub.broadcast(job_id, {
                                "type": "reviewer_feedback",
                                "feedback": final_state.get("reviewer_feedback", "")
                            })
                            
                    if next_node:
                        agent_pubsub.broadcast(job_id, {
                            "type": "node_start",
                            "node": next_node,
                            "message": msg
                        })
            
            # Save final results to DB
            job = db.query(ResearchJob).filter(ResearchJob.id == job_id).first()
            if job:
                job.status = "completed"
                job.task_list = final_state.get("task_list", [])
                job.scraped_urls = final_state.get("scraped_urls", [])
                job.citations = final_state.get("citations", {})
                job.report_draft = final_state.get("report_draft", "")
                job.revision_count = final_state.get("revision_count", 0)
                
                # Extract Title from markdown draft
                report_title = "Research Report"
                draft_lines = job.report_draft.strip().split("\n")
                if draft_lines and draft_lines[0].startswith("# "):
                    report_title = draft_lines[0].replace("# ", "").strip()
                
                # Create Report entry
                metrics = {
                    "tasks": len(job.task_list),
                    "sources": len(final_state.get("search_results", [])),
                    "pages_scraped": len(job.scraped_urls),
                    "revisions": job.revision_count,
                    "context_chunks": len(final_state.get("context_library", []))
                }
                
                report = Report(
                    job_id=job.id,
                    user_id=user_id,
                    title=report_title,
                    content=job.report_draft,
                    metrics=metrics,
                    citations=job.citations
                )
                db.add(report)
                db.commit()
                
                # Broadcast final report
                agent_pubsub.broadcast(job_id, {
                    "type": "completed",
                    "report_draft": job.report_draft,
                    "metrics": metrics,
                    "citations": job.citations
                })
                
        except Exception as e:
            # Handle failure
            import traceback
            print("Background agent flow failed:")
            traceback.print_exc()
            db.rollback()
            job = db.query(ResearchJob).filter(ResearchJob.id == job_id).first()
            if job:
                job.status = "failed"
                db.commit()
            agent_pubsub.broadcast(job_id, {
                "type": "failed",
                "error": str(e)
            })
        finally:
            db.close()
