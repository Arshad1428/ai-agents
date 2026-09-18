from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

from agent.state import AgentState

from tools.calculator import calculator
from tools.search import web_search


llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)


tools = [
    calculator,
    web_search,
]


llm_with_tools = llm.bind_tools(tools)


SYSTEM_PROMPT = """
You are a research assistant that uses tools.

You have access to these tools:

1. calculator
   Use this whenever the user asks you to perform a mathematical calculation.

2. web_search
   Use this whenever the user asks for current, recent, factual,
   or externally verifiable information.

Important:
- Do not say that you will use a tool without actually calling it.
- If web_search is needed, call web_search before answering.
- Use the information returned by the tool to formulate your answer.
- If the user asks for current information, do not rely only on your
  built-in knowledge.
"""


def agent_node(state: AgentState):

    messages = state["messages"]

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *messages,
        ]
    )

    return {
        "messages": [response]
    }


tool_node = ToolNode(tools)


builder = StateGraph(AgentState)


builder.add_node(
    "agent",
    agent_node,
)

builder.add_node(
    "tools",
    tool_node,
)


builder.add_edge(
    START,
    "agent",
)


builder.add_conditional_edges(
    "agent",
    tools_condition,
)


builder.add_edge(
    "tools",
    "agent",
)


app = builder.compile()