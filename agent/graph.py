from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from agent.state import AgentState

from tools.calculator import calculator
from tools.search import web_search


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)


# --------------------------------------------------
# Tools
# --------------------------------------------------

tools = [
    calculator,
    web_search,
]


llm_with_tools = llm.bind_tools(tools)


# --------------------------------------------------
# System prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a research assistant that uses tools.

You have access to these tools:

1. calculator
   Use this whenever the user asks you to perform
   a mathematical calculation.

2. web_search
   Use this whenever the user asks for current,
   recent, factual, or externally verifiable information.

Important rules:

- Do not say that you will use a tool without actually calling it.
- If web_search is needed, call web_search before answering.
- If calculator is needed, call calculator before answering.
- Use the information returned by the tools to formulate your answer.
- For current information, do not rely only on your built-in knowledge.
- After receiving tool results, answer the user's question clearly.
"""


# --------------------------------------------------
# Agent node
# --------------------------------------------------

def agent_node(state: AgentState):
    messages = state["messages"]

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *messages,
        ]
    )

    return {
        "messages": [response],
    }


# --------------------------------------------------
# Tool node
# --------------------------------------------------

tool_node = ToolNode(tools)


# --------------------------------------------------
# Build graph
# --------------------------------------------------

builder = StateGraph(AgentState)


builder.add_node(
    "agent",
    agent_node,
)

builder.add_node(
    "tools",
    tool_node,
)


# START → agent

builder.add_edge(
    START,
    "agent",
)


# agent → tools OR END

builder.add_conditional_edges(
    "agent",
    tools_condition,
)


# tools → agent

builder.add_edge(
    "tools",
    "agent",
)


# Compile

app = builder.compile()