from ddgs import DDGS

from langchain_core.tools import tool


MAX_RESULTS = 4
MAX_BODY_CHARS = 300
SEARCH_TIMEOUT_SECONDS = 10


@tool
def web_search(query: str) -> str:
    """Search the web and return the top results (title, snippet, URL).

    Args:
        query: A short, specific search query, for example
            "current CEO of NVIDIA" or "latest Python release".
    """
    try:
        results = DDGS(timeout=SEARCH_TIMEOUT_SECONDS).text(
            query,
            max_results=MAX_RESULTS,
        )

        if not results:
            return "No search results found."

        formatted = []

        for i, result in enumerate(results, start=1):
            body = (result.get("body") or "")[:MAX_BODY_CHARS]
            formatted.append(
                f"Result {i}\n"
                f"Title: {result.get('title', '')}\n"
                f"Content: {body}\n"
                f"URL: {result.get('href', '')}"
            )

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Web search error: {e}"
