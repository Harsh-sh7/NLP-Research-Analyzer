from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


def reviewer_node(state):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a research report reviewer. Evaluate the draft report against these criteria:\n"
            "1. Are all sub-topics covered adequately?\n"
            "2. Are sources/citations present for major claims?\n"
            "3. Is the executive summary coherent and informative?\n"
            "4. Does it follow the required markdown structure (Title, Executive Summary, "
            "Detailed Findings, Key Technical Insights, References)?\n"
            "5. Are there obvious gaps or unsupported claims?\n\n"
            "Respond with EXACTLY one of:\n"
            '- "PASS" if the report meets quality standards\n'
            '- "FAIL: [specific feedback on what needs improvement]" if it needs revision'
        )),
        ("human", (
            "Original Query: {query}\n\n"
            "Required Sub-topics: {tasks}\n\n"
            "Draft Report:\n{draft}\n\n"
            "Your verdict:"
        ))
    ])

    chain = prompt | llm
    response = chain.invoke({
        "query": state["research_query"],
        "tasks": ", ".join(state.get("task_list", [])),
        "draft": state.get("report_draft", ""),
    })

    verdict = response.content.strip()
    revision_count = state.get("revision_count", 0) + 1

    if verdict.upper().startswith("PASS"):
        return {
            "revision_count": revision_count,
            "reviewer_feedback": "",
            "status": "complete",
        }

    feedback = verdict.replace("FAIL:", "").strip() if "FAIL" in verdict.upper() else verdict
    return {
        "revision_count": revision_count,
        "reviewer_feedback": feedback,
        "status": "needs_revision",
    }
