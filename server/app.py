from fastapi import FastAPI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel

from agent.config import MAX_AGENT_STEPS
from agent.graph import app


api = FastAPI(
    title="LangGraph Research Agent",
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
# Health check
# --------------------------------------------------

@api.get("/")
def root():
    return {
        "status": "ok",
        "service": "LangGraph Research Agent",
    }


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
                "owned_by": "local",
            }
        ],
    }


# --------------------------------------------------
# Convert API messages → LangChain messages
# --------------------------------------------------

def convert_messages(messages: list[ChatMessage]):
    converted = []

    for message in messages:

        if message.role == "user":
            converted.append(
                HumanMessage(
                    content=message.content
                )
            )

        elif message.role == "assistant":
            converted.append(
                AIMessage(
                    content=message.content
                )
            )

        elif message.role == "system":
            converted.append(
                SystemMessage(
                    content=message.content
                )
            )

    return converted


# --------------------------------------------------
# OpenAI-compatible Chat Completions endpoint
# --------------------------------------------------

@api.post("/v1/chat/completions")
def chat_completions(
    request: ChatCompletionRequest,
):

    messages = convert_messages(
        request.messages
    )

    result = app.invoke(
        {
            "messages": messages,
        },
        config={
            "recursion_limit": MAX_AGENT_STEPS,
        },
    )

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