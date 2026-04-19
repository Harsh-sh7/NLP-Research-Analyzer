import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


def planner_node(state):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a research planner. Decompose the user's research query into 3-5 specific, "
            "searchable sub-questions that cover different angles of the topic.\n\n"
            "Return ONLY a JSON array of strings. No explanation, no markdown.\n"
            'Example: ["What is X?", "How does X compare to Y?", "What are recent developments in X?"]'
        )),
        ("human", "{query}")
    ])

    chain = prompt | llm
    response = chain.invoke({"query": state["research_query"]})
    content = response.content.strip()

    try:
        tasks = json.loads(content)
        if not isinstance(tasks, list):
            tasks = [state["research_query"]]
    except json.JSONDecodeError:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            try:
                tasks = json.loads(content[start:end])
            except Exception:
                tasks = [state["research_query"]]
        else:
            tasks = [state["research_query"]]

    return {
        "task_list": tasks[:5],
        "completed_tasks": [],
        "status": "planning_complete"
    }
