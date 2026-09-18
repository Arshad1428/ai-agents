from ddgs import DDGS
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """
    Search the web and return relevant results.
    """

    results = DDGS().text(
        query,
        max_results=5,
    )

    if not results:
        return "No search results found."

    formatted_results = []

    for i, result in enumerate(results, start=1):
        formatted_results.append(
            f"Result {i}\n"
            f"Title: {result.get('title', '')}\n"
            f"Content: {result.get('body', '')}\n"
            f"URL: {result.get('href', '')}"
        )

    return "\n\n".join(formatted_results)