"""API-layer tests. The LLM/agent is faked, so Ollama is not required."""
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

import server.app as server_app


class FakeAgent:
    def __init__(self, reply="The result of 25 * 48 is 1200.", error=None):
        self.reply = reply
        self.error = error
        self.calls = 0

    def invoke(self, payload, config=None):
        self.calls += 1
        if self.error:
            raise self.error
        return {"messages": [AIMessage(content=self.reply)]}


class FakePlainLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content='{"title": "Math"}')


@pytest.fixture
def client():
    return TestClient(server_app.api)


def body(content="What is 25 * 48?", **extra):
    return {
        "model": "langgraph-research-agent",
        "messages": [{"role": "user", "content": content}],
        **extra,
    }


def test_non_streaming(monkeypatch, client):
    monkeypatch.setattr(server_app, "agent_app", FakeAgent())
    r = client.post("/v1/chat/completions", json=body())
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["message"]["content"] == "The result of 25 * 48 is 1200."
    assert "created" in data and "usage" in data


def test_streaming_sse(monkeypatch, client):
    monkeypatch.setattr(server_app, "agent_app", FakeAgent("x" * 100))
    r = client.post("/v1/chat/completions", json=body(stream=True))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    lines = [l for l in r.text.split("\n\n") if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"

    chunks = [json.loads(l[6:]) for l in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "x" * 100


def test_content_as_list_of_parts(monkeypatch, client):
    agent = FakeAgent()
    monkeypatch.setattr(server_app, "agent_app", agent)
    payload = {
        "model": "m",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "tool", "content": "ignored role"},
        ],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    assert agent.calls == 1


def test_recursion_limit_becomes_friendly_message(monkeypatch, client):
    monkeypatch.setattr(server_app, "agent_app", FakeAgent(error=GraphRecursionError("x")))
    r = client.post("/v1/chat/completions", json=body())
    assert r.status_code == 200
    assert "step limit" in r.json()["choices"][0]["message"]["content"]


def test_unexpected_error_is_reported_not_500(monkeypatch, client):
    monkeypatch.setattr(server_app, "agent_app", FakeAgent(error=RuntimeError("boom")))
    r = client.post("/v1/chat/completions", json=body())
    assert r.status_code == 200
    assert "boom" in r.json()["choices"][0]["message"]["content"]


def test_background_tasks_skip_the_agent(monkeypatch, client):
    agent, plain = FakeAgent(), FakePlainLLM()
    monkeypatch.setattr(server_app, "agent_app", agent)
    monkeypatch.setattr(server_app, "plain_llm", plain)
    task = "### Task:\nGenerate a concise, 3-5 word title for the chat."
    r = client.post("/v1/chat/completions", json=body(task))
    assert r.status_code == 200
    assert agent.calls == 0 and plain.calls == 1


def test_models_endpoint(client):
    assert client.get("/v1/models").json()["data"][0]["id"] == "langgraph-research-agent"
