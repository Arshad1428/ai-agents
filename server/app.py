from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent.graph import app


api = FastAPI(
    title="LangGraph Research Agent"
)


# --------------------------------------------------
# Request models
# --------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]


# --------------------------------------------------
# OpenAI-compatible Models endpoint
# --------------------------------------------------

@api.get("/v1/models")
def models():

    return {
        "object": "list",
        "data": [
            {
                "id": "langgraph-research-agent",
                "object": "model",
                "owned_by": "local"
            }
        ]
    }


# --------------------------------------------------
# OpenAI-compatible Chat Completions endpoint
# --------------------------------------------------

@api.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):

    # Get the latest user message
    user_message = request.messages[-1].content

    # Send it to LangGraph
    result = app.invoke(
        {
            "messages": [
                    HumanMessage(content=user_message)
            ]
        }
    )

    # Get final LangGraph response
    final_message = result["messages"][-1]

    return {
        "id": "langgraph-chat-completion",
        "object": "chat.completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_message.content,
                },
                "finish_reason": "stop",
            }
        ],
    }