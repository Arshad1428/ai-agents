"""Integration test: needs a running Ollama with the configured model.

Run explicitly with:  pytest -m integration
"""
import pytest
from langchain_core.messages import HumanMessage

from agent.config import MAX_AGENT_STEPS
from agent.graph import app


@pytest.mark.integration
def test_agent_calculator():
    result = app.invoke(
        {"messages": [HumanMessage(content="What is 25 * 48?")]},
        config={"recursion_limit": MAX_AGENT_STEPS},
    )

    final_message = result["messages"][-1]
    assert "1200" in final_message.content.replace(",", "")
