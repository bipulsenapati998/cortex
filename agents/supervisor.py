from typing import Literal, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import yaml, os


# ── State definition ──────────────────────────────────────────
class CortexState(TypedDict):
    query: str
    user_id: str
    user_tier: Literal["standard", "manager", "exec"]
    intent: str  # "knowledge" | "research" | "action"
    context: list  # retrieved docs or web results
    response: str  # final answer
    cost_usd: float
    agent_used: str


# ── Load versioned prompt ─────────────────────────────────────
def load_prompt(agent: str, version: str = "v1.0.0") -> str:
    path = f"prompts/{agent}/{version}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["system_prompt"]


llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


# ── Node: Router ──────────────────────────────────────────────
def router_node(state: CortexState) -> CortexState:
    """Classify intent and determine which agent to route to."""
    system = load_prompt("supervisor")
    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(
                content=f"User tier: {state['user_tier']}\nQuery: {state['query']}"
            ),
        ]
    )
    # Parse: expect response like "intent: knowledge"
    intent = "knowledge"  # default
    for line in response.content.lower().splitlines():
        if "research" in line:
            intent = "research"
        if "action" in line:
            intent = "action"
    return {**state, "intent": intent}


# ── Routing function (conditional edge) ──────────────────────
def route_to_agent(state: CortexState) -> str:
    return state["intent"]  # "knowledge" | "research" | "action"


# ── Build the graph ───────────────────────────────────────────
def build_supervisor_graph():
    g = StateGraph(CortexState)

    g.add_node("router", router_node)
    g.add_node("knowledge", knowledge_node)  # defined in knowledge_agent.py
    g.add_node("research", research_node)  # defined in research_agent.py
    g.add_node("action", action_node)  # defined in action_agent.py

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "knowledge": "knowledge",
            "research": "research",
            "action": "action",
        },
    )
    g.add_edge("knowledge", END)
    g.add_edge("research", END)
    g.add_edge("action", END)

    return g.compile()
