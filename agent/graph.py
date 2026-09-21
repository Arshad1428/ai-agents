from datetime import date

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import (
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TIMEOUT,
)
from agent.state import AgentState

from tools.calculator import calculator
from tools.search import web_search


# --------------------------------------------------
# LLM
# --------------------------------------------------

# disable_streaming=True: avoids the streaming path that stalled for us.
# client_kwargs timeout: a stalled Ollama call raises an error instead of
# hanging forever.
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    disable_streaming=True,
    num_ctx=OLLAMA_NUM_CTX,
    num_predict=OLLAMA_NUM_PREDICT,
    keep_alive=OLLAMA_KEEP_ALIVE,
    client_kwargs={"timeout": OLLAMA_TIMEOUT},
)

tools = [
    calculator,
    web_search,
]

llm_with_tools = llm.bind_tools(tools)

# Plain model without tools. Used for background tasks such as chat title
# generation, so they never trigger web searches.
plain_llm = llm


# --------------------------------------------------
# System prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a research assistant with two tools.

Today's date is {today}.

- calculator: use for any arithmetic.
- web_search: use for current, recent, or externally verifiable facts.

Rules:
- If a tool is needed, call it before answering. Never say you will use a
  tool without calling it.
- Base your answer on the tool results. Prefer the most recent dated
  information, and do not fall back on older values from memory.
- Do not invent facts. If the results do not clearly answer the question,
  say the available results are insufficient.
- Do not repeat the same search. Try at most one clearly different query,
  then answer with what you have.
- Be concise.
"""


# --------------------------------------------------
# Agent node
# --------------------------------------------------

def agent_node(state: AgentState):
    system = SystemMessage(
        content=SYSTEM_PROMPT.format(today=date.today().isoformat())
    )

    response = llm_with_tools.invoke([system, *state["messages"]])

    return {"messages": [response]}


# --------------------------------------------------
# Build graph
# --------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

app = builder.compile()
