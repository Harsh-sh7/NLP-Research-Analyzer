from agent.tools.scrape_tool import scrape_url, chunk_text, retrieve_relevant_chunks


def scraper_node(state):
    search_results = state.get("search_results", [])
    task_list = state.get("task_list", [])
    already_scraped = set(state.get("scraped_urls", []))

    urls = []
    for r in search_results:
        url = r.get("url", "")
        if url and url not in already_scraped:
            urls.append(url)
            already_scraped.add(url)
        if len(urls) >= 6:
            break

    new_context = []
    new_citations = {}

    for url in urls:
        text = scrape_url(url)
        if not text or len(text) < 100:
            continue

        chunks = chunk_text(text)

        for task in task_list:
            relevant = retrieve_relevant_chunks(chunks, task, k=2)
            for chunk in relevant:
                tagged = f"[Source: {url}]\n{chunk}"
                new_context.append(tagged)
                key = chunk[:100]
                new_citations[key] = url

    return {
        "context_library": new_context,
        "citations": {**state.get("citations", {}), **new_citations},
        "scraped_urls": list(already_scraped),
        "status": "scraping_complete"
    }
