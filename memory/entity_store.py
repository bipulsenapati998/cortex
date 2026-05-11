"""
memory/entity_store.py
PostgreSQL entity store — long-term facts that survive server restarts.
Week 3 pattern — durable entity memory.
"""

import os
import json
import logging
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import LLM_MODEL

logger = logging.getLogger("cortex.entity_store")

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return _llm

# ── Entity extraction ──────────────────────────────────────────────────────

def extract_entities(text: str) -> Dict[str, str]:
    """
    Use LLM to extract named entities from a user message.
    Returns a dict like {"name": "Riya", "order_id": "ORD-789"}.
    Falls back to {} on any error.
    """
    try:
        llm = _get_llm()
        resp = llm.invoke([
            SystemMessage(content=(
                "Extract named entities from the user message. "
                "Return ONLY valid JSON in this format: "
                "{\"entities\": {\"key\": \"value\"}}. "
                "Keys should be snake_case (e.g. order_id, name, department). "
                "If no entities found, return {\"entities\": {}}."
            )),
            HumanMessage(content=text),
        ])
        raw = resp.content.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw).get("entities", {})
    except Exception as e:
        logger.debug("[EntityStore] extract_entities failed: %s", e)
        return {}


# ── PostgreSQL persistence ─────────────────────────────────────────────────

def upsert_entities(user_id: str, entities: Dict[str, str]) -> None:
    """Insert or update entities for a user in PostgreSQL."""
    if not entities:
        return
    try:
        import psycopg2
        with psycopg2.connect(os.getenv("POSTGRES_URL", "")) as conn:
            with conn.cursor() as cur:
                for key, val in entities.items():
                    cur.execute(
                        """
                        INSERT INTO user_entities (user_id, entity_key, entity_val)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (user_id, entity_key)
                        DO UPDATE SET entity_val = EXCLUDED.entity_val,
                                      updated_at = NOW()
                        """,
                        (user_id, key, str(val)),
                    )
    except Exception as e:
        logger.warning("[EntityStore] upsert_entities failed: %s", e)


def load_entities(user_id: str) -> Dict[str, str]:
    """Load all stored entities for a user."""
    try:
        import psycopg2
        with psycopg2.connect(os.getenv("POSTGRES_URL", "")) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT entity_key, entity_val FROM user_entities WHERE user_id = %s",
                    (user_id,),
                )
                return {row[0]: row[1] for row in cur.fetchall()}
    except Exception as e:
        logger.warning("[EntityStore] load_entities failed: %s", e)
        return {}

def format_entity_context(entities: Dict[str, str]) -> str:
    """Format entity dict into a prompt-injectable context string."""
    if not entities:
        return ""
    lines = [f"  - {k}: {v}" for k, v in entities.items()]
    return "Known user context:\n" + "\n".join(lines)
