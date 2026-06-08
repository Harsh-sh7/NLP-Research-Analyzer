import operator
from typing import TypedDict, Annotated


class AgentState(TypedDict):
    research_query: str
    task_list: list[str]
    completed_tasks: list[str]
    search_results: list[dict]
    scraped_urls: list[str]
    context_library: Annotated[list[str], operator.add]
    citations: dict[str, str]
    report_draft: str
    revision_count: int
    reviewer_feedback: str
    status: str
