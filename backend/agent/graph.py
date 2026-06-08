from langgraph.graph import StateGraph, END
from backend.agent.state import AgentState
from backend.agent.nodes.planner import planner_node
from backend.agent.nodes.researcher import researcher_node
from backend.agent.nodes.scraper import scraper_node
from backend.agent.nodes.synthesizer import synthesizer_node
from backend.agent.nodes.reviewer import reviewer_node


def review_decision(state):
    if state.get("status") == "complete":
        return "end"
    if state.get("revision_count", 0) >= 2:
        return "end"
    return "researcher"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("scraper", scraper_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "scraper")
    graph.add_edge("scraper", "synthesizer")
    graph.add_edge("synthesizer", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        review_decision,
        {"researcher": "researcher", "end": END}
    )

    return graph.compile()
