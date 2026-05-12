import re
import logging
from typing import Any, Literal, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import yaml, os

from config import SUPERVISOR_PROMPT_VERSION, LLM_MODEL
from prompts.prompt_loader import load_prompt as _loader_load_prompt
from knowledge_agent import knowledge_node

logger = logging.getLogger("cortex.supervisor")

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
    budget: Any  # QueryBudget instance
    # RBAC fields
    rbac_context: Optional[Any]  # RBACContext | None 
    user_role: str
    user_name: str
    auth_error: Optional[str]
    # Auth
    auth_token: str

# ── Prompt injection defence ───────────────────────────────────────────────

_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(?:all\s+)?(?:previous\s+)?instructions|"
    r"disregard\s+(?:your\s+)?(?:previous\s+)?instructions|"
    r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|another)|"
    r"act\s+as\s+(?:a\s+)?(?:different|new|another)|"
    r"forget\s+(?:all\s+)?(?:your\s+)?(?:previous\s+)?(?:instructions|training)|"
    r"override\s+(?:your\s+)?(?:previous\s+)?(?:instructions|constraints)|"
    r"jailbreak|DAN\s+mode|developer\s+mode|unrestricted\s+mode)",
    re.IGNORECASE,
)


def sanitise_input(query: str) -> str:
    """
    Detect and neutralise prompt injection attempts.
    Returns cleaned query or a safe placeholder.
    """
    if _INJECTION_PATTERNS.search(query):
        logger.warning("[Supervisor] Prompt injection attempt detected in query: %r", query[:80])
        return "[SECURITY: Potentially unsafe input was sanitised]"
    # Truncate excessively long queries (token safety)
    return query[:2000]


# ── Load versioned prompt ──────────────────────────────────────────────
def load_prompt(agent: str = "supervisor", version: str = "v1.0.0") -> str:
    """
    Load the versioned system prompt for the Supervisor.
    """
    v = version or SUPERVISOR_PROMPT_VERSION
    return _loader_load_prompt(agent, version=v, section="system_prompt")

# ── LLM ───────────────────────────────────────────────────────────────────

def _get_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=0)


# ── Node: Router ──────────────────────────────────────────────
# def auth_node(state: CortexState) -> CortexState:
#     """Entry node: verify JWT and inject RBAC context into state."""
#     token = state.get("auth_token", "")
#     if not token:
#         # Development mode: map user_tier to a role directly
#         tier_to_role = {"standard": "support_agent", "manager": "analyst", "exec": "admin"}
#         role = tier_to_role.get(state.get("user_tier", "standard"), "support_agent")
#         return {
#             **state,
#             "user_role":    role,
#             "user_name":    state.get("user_id", "anonymous"),
#             "auth_error":   None,
#             "rbac_context": None,
#         }
#     return inject_rbac_into_state(state, token)

def router_node(state: CortexState) -> CortexState:
    """
    Classify intent and determine which agent to route to.
    Enforce tier-based routing rules.
    Uses versioned prompt loaded from YAML.
    """
    # if state.get("auth_error"):
    #     return {**state, "intent": "fallback", "response": "Authentication failed. Please log in again."}
    system = load_prompt("supervisor", version=SUPERVISOR_PROMPT_VERSION)
    llm = _get_llm()
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
