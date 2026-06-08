import os
import sys
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Any

# Add workspace root to sys.path so LangGraph agent code can be imported
WORKSPACE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if WORKSPACE_ROOT not in sys.path:
    sys.path.append(WORKSPACE_ROOT)

from backend.agent.graph import build_graph
from backend.app.db.session import get_client
from backend.app.core.config import settings


class AgentPubSub:
    def __init__(self):
        self._listeners: Dict[str, List[asyncio.Queue]] = {}

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self._listeners.setdefault(job_id, []).append(q)
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
        for q in self._listeners.get(job_id, []):
            q.put_nowait(data)


agent_pubsub = AgentPubSub()


class AgentService:
    @staticmethod
    def start_research_task(job_id: str, query: str, user_id: str):
        """Schedule the research agent as a background asyncio Task."""
        asyncio.create_task(AgentService._run_agent_flow(job_id, query, user_id))

    @staticmethod
    async def _run_agent_flow(job_id: str, query: str, user_id: str):
        """Run the LangGraph pipeline and persist all state changes to MongoDB."""
        db = get_client()[settings.MONGODB_DB_NAME]

        try:
            # Mark job as running
            db["research_jobs"].update_one(
                {"_id": job_id},
                {"$set": {"status": "running"}}
            )

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
            agent_pubsub.broadcast(job_id, {
                "type": "node_start",
                "node": "planner",
                "message": "Planner node active. Decomposing research query...",
            })

            loop = asyncio.get_running_loop()

            def get_stream_iterator():
                return agent.stream(initial_state)

            def get_next_event(it):
                try:
                    return next(it)
                except StopIteration:
                    return None

            iterator = await loop.run_in_executor(None, get_stream_iterator)

            final_state: Dict[str, Any] = {}
            iteration = 0

            while True:
                event = await loop.run_in_executor(None, get_next_event, iterator)
                if event is None:
                    break

                for node_name, node_output in event.items():
                    if node_name.startswith("__"):
                        continue

                    if isinstance(node_output, dict):
                        final_state.update(node_output)

                    agent_pubsub.broadcast(job_id, {
                        "type": "node_complete",
                        "node": node_name,
                        "data": {
                            "task_list": final_state.get("task_list", []),
                            "scraped_urls": final_state.get("scraped_urls", []),
                            "search_results_count": len(final_state.get("search_results", [])),
                            "revision_count": final_state.get("revision_count", 0),
                            "status": final_state.get("status", ""),
                        },
                    })

                    next_node = None
                    msg = ""
                    if node_name == "planner":
                        next_node, msg = "researcher", "Researcher node active. Searching web..."
                    elif node_name == "researcher":
                        count = len(final_state.get("search_results", []))
                        next_node, msg = "scraper", f"Scraper node active. Indexing {count} results..."
                    elif node_name == "scraper":
                        next_node, msg = "synthesizer", "Synthesizer node active. Drafting markdown report..."
                    elif node_name == "synthesizer":
                        next_node, msg = "reviewer", "Reviewer node active. Assessing quality parameters..."
                    elif node_name == "reviewer":
                        if (
                            final_state.get("status") == "needs_revision"
                            and final_state.get("revision_count", 0) < 2
                        ):
                            iteration += 1
                            next_node = "researcher"
                            msg = f"Revision cycle {iteration} triggered. Querying search engines again..."
                            agent_pubsub.broadcast(job_id, {
                                "type": "reviewer_feedback",
                                "feedback": final_state.get("reviewer_feedback", ""),
                            })

                    if next_node:
                        agent_pubsub.broadcast(job_id, {
                            "type": "node_start",
                            "node": next_node,
                            "message": msg,
                        })

            # ── Persist completed state ────────────────────────────────────────
            task_list = final_state.get("task_list", [])
            scraped_urls = final_state.get("scraped_urls", [])
            citations = final_state.get("citations", {})
            report_draft = final_state.get("report_draft", "")
            revision_count = final_state.get("revision_count", 0)

            db["research_jobs"].update_one(
                {"_id": job_id},
                {"$set": {
                    "status": "completed",
                    "task_list": task_list,
                    "scraped_urls": scraped_urls,
                    "citations": citations,
                    "report_draft": report_draft,
                    "revision_count": revision_count,
                }}
            )

            # Extract title from markdown
            report_title = "Research Report"
            draft_lines = report_draft.strip().split("\n")
            if draft_lines and draft_lines[0].startswith("# "):
                report_title = draft_lines[0].replace("# ", "").strip()

            metrics = {
                "tasks": len(task_list),
                "sources": len(final_state.get("search_results", [])),
                "pages_scraped": len(scraped_urls),
                "revisions": revision_count,
                "context_chunks": len(final_state.get("context_library", [])),
            }

            report_doc = {
                "_id": str(uuid.uuid4()),
                "job_id": job_id,
                "user_id": user_id,
                "title": report_title,
                "content": report_draft,
                "metrics": metrics,
                "citations": citations,
                "created_at": datetime.utcnow(),
            }
            db["reports"].insert_one(report_doc)

            agent_pubsub.broadcast(job_id, {
                "type": "completed",
                "report_draft": report_draft,
                "metrics": metrics,
                "citations": citations,
            })

        except Exception as e:
            import traceback
            print("Background agent flow failed:")
            traceback.print_exc()

            db["research_jobs"].update_one(
                {"_id": job_id},
                {"$set": {"status": "failed"}}
            )
            agent_pubsub.broadcast(job_id, {"type": "failed", "error": str(e)})
