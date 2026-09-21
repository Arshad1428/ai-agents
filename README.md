# ai-agents

A local LangGraph research agent (Ollama + calculator + web search) exposed
as an OpenAI-compatible API so Open WebUI can use it.

```
Open WebUI -> FastAPI (server/app.py) -> LangGraph (agent/graph.py) -> Ollama
                                               |-> calculator
                                               |-> web_search (DuckDuckGo)
```

## Setup (Windows / Git Bash)

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env          # then edit if needed
ollama pull qwen2.5:7b-instruct
```

## Run the API

```bash
uvicorn server.app:api --host 0.0.0.0 --port 8000
```

Quick check (the first call can take 15-20s while the model loads):

```bash
curl -s --max-time 120 http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"langgraph-research-agent","messages":[{"role":"user","content":"What is 25 * 48?"}]}'
```

## Connect Open WebUI

Admin Settings -> Connections -> add an OpenAI API connection:

- URL: `http://localhost:8000/v1` (use `http://host.docker.internal:8000/v1`
  if Open WebUI runs in Docker)
- API key: any non-empty value

Then pick the `langgraph-research-agent` model.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Model used by the agent |
| `MAX_AGENT_STEPS` | `15` | LangGraph step limit (about 2-3 steps per tool call) |
| `OLLAMA_TIMEOUT` | `180` | Seconds before a stalled Ollama call errors out |
| `OLLAMA_NUM_CTX` | `8192` | Context window (fixed so Ollama can't pick a huge default) |
| `OLLAMA_NUM_PREDICT` | `768` | Max tokens generated per model call |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long the model stays loaded after a request (`-1` = forever) |
| `WARMUP_ON_START` | `true` | Load the model when the API starts |

## Speed

Speed is mostly decided by model size and whether it runs on a GPU. Check
with `ollama ps` while a request is running: the PROCESSOR column shows
`100% GPU` (good) or `100% CPU` (slow for 7B models).

Measure models on your own machine (pull them first):

```bash
python -m scripts.benchmark                                # 3 default models
python -m scripts.benchmark qwen2.5:7b-instruct llama3.2   # or pick your own
```

It reports load time, the tool-call turn and the answer turn per model, and
whether the model actually produced a tool call. Pick the fastest model that
gets `tool call OK`, then set `OLLAMA_MODEL` in `.env`.

Optional Ollama *server* settings (set as environment variables, then restart
Ollama): `OLLAMA_NUM_PARALLEL=1` (single user, so don't reserve memory for 4
parallel requests) and `OLLAMA_FLASH_ATTENTION=1`.

To go back to the previous model, change one line in `.env`:
`OLLAMA_MODEL=llama3.2`.

## Tests

```bash
pytest                    # unit + API tests, no Ollama needed
pytest -m integration     # real agent test, needs Ollama running
```

## Notes

- `ChatOllama` is created with `disable_streaming=True` (the streaming path
  stalled during development) and a client timeout, so a stuck call becomes
  an error instead of a silent hang.
- The API supports both normal and `stream: true` responses. The agent runs
  to completion first, then the answer is streamed out in chunks.
- Open WebUI's background requests (chat titles, tags, follow-ups) start with
  `### Task:` and are answered by the plain model without tools.
- `qwen2.5:7b-instruct` stalled on the original machine before the timeout
  and settings above existed. If it misbehaves, run the benchmark and fall
  back to `llama3.2` or `qwen2.5:3b-instruct`.
