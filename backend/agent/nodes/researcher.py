import time
from backend.agent.tools.search_tool import web_search


def researcher_node(state):
    all_results = []
    tasks = state.get("task_list", [])
    feedback = state.get("reviewer_feedback", "")

    if feedback:
        tasks = tasks + [f"{state['research_query']} {feedback}"]

    for task in tasks:
        results = web_search(task, max_results=5)
        all_results.extend(results)
        time.sleep(0.5)

    return {
        "search_results": all_results,
        "status": "research_complete"
    }
