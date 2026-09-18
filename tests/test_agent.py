from langchain_core.messages import HumanMessage

from agent.graph import app


result = app.invoke(
    {
        "messages": [
            HumanMessage(content="Who is the current CEO of Microsoft?")
        ]
    }
)


final_message = result["messages"][-1]

print(final_message.content)