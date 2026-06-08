import os
import time
from langchain_tavily import TavilySearch


def web_search(query: str, max_results: int = 5) -> list[dict]:
    try:
        tool = TavilySearch(max_results=max_results, topic="general")
        raw = tool.invoke({"query": query})
        
        # Handle dict response from newer versions of langchain-tavily
        raw_list = raw.get("results", []) if isinstance(raw, dict) else raw
        
        results = []
        for r in raw_list:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            })
        return results
    except Exception as e:
        print(f"Search failed for '{query}': {e}")
        time.sleep(2)
        try:
            tool = TavilySearch(max_results=max_results, topic="general")
            raw = tool.invoke({"query": query})
            raw_list = raw.get("results", []) if isinstance(raw, dict) else raw
            return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")} for r in raw_list]
        except Exception:
            return []
