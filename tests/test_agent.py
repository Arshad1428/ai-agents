from langchain_core.messages import HumanMessage

from agent.config import MAX_AGENT_STEPS
from agent.graph import app


def test_agent():
    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is 25 * 48?"
                )
            ]
        },
        config={
            "recursion_limit": MAX_AGENT_STEPS,
        },
    )

    final_message = result["messages"][-1]

    print("\nAgent response:")
    print(final_message.content)

    assert final_message.content