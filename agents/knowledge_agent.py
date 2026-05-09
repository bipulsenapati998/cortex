"""
Knowledge Agent — specialist for internal document Q&A.
"""

import logging
import yaml
from langchain_openai import ChatOpenAI

from config import LLM_MODEL, OPEN_API_KEY
from memory.session_memory import SessionMemory

logger = logging.getLogger("cortex.knowledge_agent")

_session_memory = SessionMemory()
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return _llm
