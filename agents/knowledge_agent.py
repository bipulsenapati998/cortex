"""
Knowledge Agent — specialist for internal document Q&A.
"""

import logging
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import KNOWLEDGE_AGENT_PROMPT_VERSION, LLM_MODEL, OPEN_API_KEY
from memory.session_memory import SessionMemory
from observability.logger import log_retrieval
from prompts.prompt_loader import load_prompt
from reliability.cost_tracker import QueryBudget
from rag.query_understanding import understand_query
from rag.hybrid_search import hybrid_search
from access_control import filter_by_tier
from memory.entity_store import extract_entities, format_entity_context, load_entities, upsert_entities

logger = logging.getLogger("cortex.knowledge_agent")

_session_memory = SessionMemory()
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return _llm


def _load_prompt(version: str = None):
    """
    Load the versioned system prompt for the Knowledge Agent.

    Uses prompt_loader.load_prompt() which reads:
      prompts/agents/knowledge_agent/{version}.yaml → system_prompt section

    The YAML has 5 sections:
      - role_and_constraints   (who the agent is + hard rules)
      - context_and_examples   (few-shot format examples)
      - top_guard              (security preamble — injected at TOP)
      - bottom_guard           (self-check checklist — injected at BOTTOM)
      - system_prompt          (assembled full prompt used at runtime)

    Falls back to a hardcoded default string if the file is missing.

    Returns:
        str : the full system prompt string ready for SystemMessage()
    """

    ver = version or KNOWLEDGE_AGENT_PROMPT_VERSION
    prompt = load_prompt("knowledge_agent", version=ver, section="system_prompt")
    logger.debug(
        "[KnowledgeAgent] Loaded prompt version=%s (%d chars)", ver, len(prompt)
    )
    return prompt

def _build_context_string(chunks: list) -> str:
    """Format retrieved chunks into the [Doc N: source] block the prompt expects."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta   = chunk.get("metadata", {})
        source = meta.get("source") or meta.get("doc_id") or chunk.get("doc_id", "unknown")
        title  = meta.get("title", "")
        label  = f"{source}" + (f" — {title}" if title else "")
        parts.append(f"[Doc {i}: {label}]\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)

def knowledge_node(state: dict) -> dict:
    """
    LangGraph node — Knowledge Agent.

    Wiring of _load_prompt():
      The versioned system prompt (loaded from YAML) is injected as the
      first SystemMessage. This means:
        • top_guard    — is the first thing the LLM reads (security boundary)
        • role_and_constraints — defines behaviour
        • context_and_examples — teaches citation format
        • bottom_guard — is the last thing before the user turn (self-check)
      All of these live in system_prompt inside the YAML and are loaded here.
    """
    query_text = state.get("query", "")
    user_id = state.get("user_id", "anonymous")
    user_tier = state.get("user_tier", "standard")
    budget: QueryBudget = state.get("budget", QueryBudget())

    # ── 1. Load versioned system prompt ──────────────────────────────────
    system_prompt_text = _load_prompt()  # THIS is where _load_prompt() is used

    # ── 2. Load entity context (persistent facts about this user) ────────
    entities = load_entities(user_id)
    entity_ctx = format_entity_context(entities)

    # ── 3. Load session history (last 6 turns from Redis) ───────────────
    session_history = _session_memory.load(user_id)

    # ── 4. Run RAG retrieval (Layers 5-7) ────────────────────────────────
    try:
        understood = understand_query(query_text)
        expanded = understood["expanded"]
        raw_results = hybrid_search(expanded, k=10)
        filtered = filter_by_tier(raw_results, user_tier)
        chunks = filtered[:5]
        log_retrieval(query_text, expanded, len(chunks), user_tier)
    except Exception as e:
        logger.error("[KnowledgeAgent] Retrieval error: %s", e)
        chunks = []

    # ── 5. Call LLM with versioned prompt + context + history + query ────
    api_key = OPEN_API_KEY
    if not api_key or api_key.startswith("sk-...") or not chunks:
        # Offline fallback or empty results
        if chunks:
            answer = f"Based on our knowledge base: {chunks[0]['content'][:500]}"
        else:
            answer = (
                "I couldn't find relevant information in the knowledge base for your query. "
                "Please contact hr@novatech.co or itsupport@novatech.com directly."
            )
        sources = chunks
    else:
        context_str = _build_context_string(chunks)

        # Inject entity context into system prompt if available
        full_system = system_prompt_text
        if entity_ctx:
            full_system += (
                f"\n\n---\n\nUSER CONTEXT (from prior sessions):\n{entity_ctx}"
            )

        # Build message list:
        # [SystemMessage(versioned_prompt)] + session_history[-4:] + [HumanMessage(context+query)]
        messages = [SystemMessage(content=full_system)]
        if session_history:
            messages.extend(session_history[-4:])  # last 4 messages = 2 turns
        messages.append(
            HumanMessage(
                content=(
                    f"Context documents:\n{context_str}\n\n" f"Question: {query_text}"
                )
            )
        )

        try:
            llm = _get_llm()
            resp = llm.invoke(messages)
            answer = resp.content
            budget.add(tokens=500, model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        except Exception as e:
            logger.error("[KnowledgeAgent] LLM generation failed: %s", e)
            answer = (
                f"I found {len(chunks)} relevant document(s) but could not generate "
                f"a response right now. Please try again."
            )
        sources = chunks

    # ── 6. Extract & persist entities from this turn ─────────────────────
    try:
        new_entities = extract_entities(query_text)
        if new_entities:
            upsert_entities(user_id, new_entities)
    except Exception:
        pass

    # ── 7. Save turn to session memory ───────────────────────────────────
    try:
        _session_memory.append_and_save(user_id, query_text, answer)
    except Exception:
        pass

    logger.info(
        "[KnowledgeAgent] user=%s tier=%s chunks=%d prompt_ver=%s budget=%s",
        user_id,
        user_tier,
        len(sources),
        PROMPT_VERSION,
        budget.summary(),
    )

    return {
        **state,
        "response": answer,
        "context": sources,
        "agent_used": "knowledge",
        "cost_usd": budget.spent,
    }
