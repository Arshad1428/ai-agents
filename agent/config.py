import os

from dotenv import load_dotenv


load_dotenv()


def _keep_alive(value: str):
    """Ollama accepts durations like '30m' or a number of seconds (-1 = forever)."""
    try:
        return int(value)
    except ValueError:
        return value


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# LangGraph counts *graph steps*, not tool calls. One tool round trip
# (agent -> tools -> agent) is about 2-3 steps, so 15 allows ~5 tool calls.
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "15"))

# Seconds to wait for Ollama before failing with an error instead of hanging.
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))

# --- Speed settings -------------------------------------------------
# Context window. Fixed explicitly so Ollama never picks a huge default
# (bigger context = more memory and slower loads).
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

# Upper bound on generated tokens per model call. Caps worst-case latency.
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "768"))

# How long Ollama keeps the model in memory after a request. Avoids the
# 15-30s cold load after idle periods (Ollama's own default is 5 minutes).
OLLAMA_KEEP_ALIVE = _keep_alive(os.getenv("OLLAMA_KEEP_ALIVE", "30m"))

# Load the model into memory when the API starts, so the first real
# question isn't slow.
WARMUP_ON_START = os.getenv("WARMUP_ON_START", "true").lower() in {"1", "true", "yes"}
