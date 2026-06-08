from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


def synthesizer_node(state):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

    context = "\n\n".join(state.get("context_library", []))
    snippets = "\n".join([
        f"- {r.get('title', '')}: {r.get('snippet', '')}"
        for r in state.get("search_results", [])
    ])

    feedback = state.get("reviewer_feedback", "")
    feedback_block = ""
    if feedback:
        feedback_block = f"\n\nREVIEWER FEEDBACK TO ADDRESS:\n{feedback}"

    urls = list(set(
        r.get("url", "") for r in state.get("search_results", []) if r.get("url")
    ))
    urls_block = "\n".join([f"- {u}" for u in urls[:15]])

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a senior research synthesizer. Write a comprehensive, well-structured "
            "research report using ONLY the provided context and search data.\n\n"
            "Your report MUST follow this EXACT markdown structure:\n\n"
            "# [Research Topic Title]\n\n"
            "## Executive Summary\n"
            "A 2-3 paragraph overview of key findings.\n\n"
            "## Detailed Findings\n"
            "### [Sub-topic 1]\n...\n"
            "### [Sub-topic 2]\n...\n\n"
            "## Key Technical Insights\n"
            "Bullet points of the most critical data, metrics, or facts.\n\n"
            "## References\n"
            "A numbered list of source URLs.\n\n"
            "RULES:\n"
            "- Cite sources inline using [Source: URL]\n"
            "- If info is unavailable, state the limitation clearly\n"
            "- Maintain a neutral, professional tone\n"
            "- Do NOT hallucinate or invent facts beyond the provided data"
        )),
        ("human", (
            "Research Query: {query}\n\n"
            "Sub-topics to cover: {tasks}\n\n"
            "Search Results:\n{snippets}\n\n"
            "Scraped Context:\n{context}\n\n"
            "Available Source URLs:\n{urls}{feedback}\n\n"
            "Write the complete research report now."
        ))
    ])

    chain = prompt | llm
    response = chain.invoke({
        "query": state["research_query"],
        "tasks": ", ".join(state.get("task_list", [])),
        "snippets": snippets[:4000],
        "context": context[:12000],
        "urls": urls_block,
        "feedback": feedback_block,
    })

    return {
        "report_draft": response.content,
        "status": "synthesis_complete"
    }
