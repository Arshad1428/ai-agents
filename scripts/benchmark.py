"""Compare Ollama models on THIS machine, using the same settings as the agent.

Run from the project root (with the venv active):

    python -m scripts.benchmark                       # default model list
    python -m scripts.benchmark qwen2.5:7b-instruct llama3.2

For each model it measures: load time, a tool-call turn, and the follow-up
turn that turns tool output into an answer. Models must be pulled first
(`ollama pull <name>`).
"""
import sys
import time

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

from agent.config import (
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
)
from tools.calculator import calculator
from tools.search import web_search

DEFAULT_MODELS = ["qwen2.5:3b-instruct", "qwen2.5:7b-instruct", "llama3.2"]
QUESTION = HumanMessage(content="What is 25 * 48?")


def make_llm(model):
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        disable_streaming=True,
        num_ctx=OLLAMA_NUM_CTX,
        num_predict=OLLAMA_NUM_PREDICT,
        keep_alive=OLLAMA_KEEP_ALIVE,
        client_kwargs={"timeout": 240},
    )


def tokens_per_second(message):
    meta = message.response_metadata or {}
    count, nanos = meta.get("eval_count"), meta.get("eval_duration")
    return count / (nanos / 1e9) if count and nanos else None


def timed(fn):
    start = time.time()
    result = fn()
    return result, time.time() - start


def bench(model):
    print(f"\n=== {model} ===", flush=True)
    llm = make_llm(model)
    llm_tools = llm.bind_tools([calculator, web_search])

    _, load_s = timed(lambda: llm.invoke("Reply with the single word: ready"))
    print(f"  load + first reply : {load_s:5.1f}s", flush=True)

    ai, call_s = timed(lambda: llm_tools.invoke([QUESTION]))
    ok = any(c["name"] == "calculator" for c in ai.tool_calls)
    tps = tokens_per_second(ai)
    print(
        f"  tool-call turn     : {call_s:5.1f}s"
        f"  tool call {'OK' if ok else 'MISSING'}"
        + (f"  ({tps:.1f} tok/s)" if tps else ""),
        flush=True,
    )
    if not ok:
        print(f"  (model replied with text instead: {ai.content!r})", flush=True)
        return {"model": model, "load": load_s, "call": call_s, "answer": None, "ok": False}

    call = ai.tool_calls[0]
    result = calculator.invoke(call["args"])
    final, answer_s = timed(
        lambda: llm_tools.invoke(
            [QUESTION, ai, ToolMessage(content=result, tool_call_id=call["id"])]
        )
    )
    print(f"  answer turn        : {answer_s:5.1f}s  -> {final.content.strip()!r}", flush=True)
    return {"model": model, "load": load_s, "call": call_s, "answer": answer_s, "ok": True}


def main():
    models = sys.argv[1:] or DEFAULT_MODELS
    rows = []

    for model in models:
        try:
            rows.append(bench(model))
        except Exception as error:
            print(f"  FAILED: {type(error).__name__}: {error}", flush=True)
            print(f"  (is it pulled?  ollama pull {model})", flush=True)

    print("\n=== Summary (a full tool question ~ tool-call turn + answer turn) ===")
    for r in rows:
        if r["ok"]:
            print(f"  {r['model']:<24} ~{r['call'] + r['answer']:5.1f}s per tool question (warm)")
        else:
            print(f"  {r['model']:<24} did not produce a tool call")


if __name__ == "__main__":
    main()
