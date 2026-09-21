import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, ConfigDict

from agent.config import MAX_AGENT_STEPS, WARMUP_ON_START
from agent.graph import app as agent_app
from agent.graph import plain_llm


# Same logger uvicorn uses, so INFO messages show up in the terminal.
logger = logging.getLogger("uvicorn.error")

MODEL_ID = "langgraph-research-agent"

# Open WebUI sends its background jobs (chat title, tags, follow-up
# suggestions, search-query generation) as a single user message that
# starts with this marker. They must not go through the tool agent.
BACKGROUND_TASK_MARKER = "### Task:"

STREAM_CHUNK_CHARS = 24


def warm_up():
    """Load the model into memory so the first real question is fast."""
    try:
        started = time.time()
        plain_llm.invoke("Reply with the single word: ready")
        logger.info("Model warm-up finished in %.1fs.", time.time() - started)
    except Exception as error:
        # Ollama may not be up yet; the first real request will load the model.
        logger.warning("Model warm-up skipped: %s", error)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if WARMUP_ON_START:
        # Background thread: the API accepts requests immediately.
        threading.Thread(target=warm_up, daemon=True).start()
    yield


api = FastAPI(title="LangGraph Research Agent", lifespan=lifespan)


# --------------------------------------------------
# Request models
# --------------------------------------------------

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    # OpenAI clients may send a plain string, a list of content parts,
    # or null (e.g. assistant messages that only contain tool calls).
    content: str | list[Any] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False


# --------------------------------------------------
# Health / models
# --------------------------------------------------

@api.get("/")
def root():
    return {"status": "ok", "service": "LangGraph Research Agent"}


@api.get("/health")
def health():
    return {"status": "ok"}


@api.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def text_from_content(content: Any) -> str:
    """Flatten str / list-of-parts / None message content into plain text."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    parts = []

    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))

    return "\n".join(p for p in parts if p)


def convert_messages(messages: list[ChatMessage]):
    """API messages -> LangChain messages. Unsupported roles are skipped."""
    converted = []

    for message in messages:
        text = text_from_content(message.content)

        if message.role == "user":
            converted.append(HumanMessage(content=text))
        elif message.role == "assistant":
            converted.append(AIMessage(content=text))
        elif message.role == "system":
            converted.append(SystemMessage(content=text))

    return converted


def is_background_task(messages: list[ChatMessage]) -> bool:
    for message in reversed(messages):
        if message.role == "user":
            text = text_from_content(message.content)
            return text.lstrip().startswith(BACKGROUND_TASK_MARKER)

    return False


def run_agent(messages: list[ChatMessage]) -> str:
    """Run the request and always return text (never raises)."""
    langchain_messages = convert_messages(messages)

    if not langchain_messages:
        return "No messages were provided."

    background = is_background_task(messages)

    try:
        if background:
            reply = plain_llm.invoke(langchain_messages)
            return text_from_content(reply.content).strip()

        result = agent_app.invoke(
            {"messages": langchain_messages},
            config={"recursion_limit": MAX_AGENT_STEPS},
        )

        text = text_from_content(result["messages"][-1].content).strip()
        return text or "The agent returned an empty response."

    except GraphRecursionError:
        logger.warning("Agent hit the step limit (%s).", MAX_AGENT_STEPS)
        return (
            "I hit my step limit before finishing. Please try a more "
            "specific question."
        )

    except Exception as error:
        logger.exception("Agent run failed")

        if background:
            return ""

        return f"Agent error: {type(error).__name__}: {error}"


def sse(chunk_id: str, created: int, model: str, delta: dict, finish=None):
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish}
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def stream_reply(request: ChatCompletionRequest):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = request.model

    # Send the role immediately so the client sees the response start
    # while the (slow) agent is still working.
    yield sse(chunk_id, created, model, {"role": "assistant", "content": ""})

    text = run_agent(request.messages)

    for start in range(0, len(text), STREAM_CHUNK_CHARS):
        yield sse(
            chunk_id,
            created,
            model,
            {"content": text[start : start + STREAM_CHUNK_CHARS]},
        )

    yield sse(chunk_id, created, model, {}, finish="stop")
    yield "data: [DONE]\n\n"


# --------------------------------------------------
# OpenAI-compatible Chat Completions
# --------------------------------------------------

@api.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        return StreamingResponse(
            stream_reply(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    text = run_agent(request.messages)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
