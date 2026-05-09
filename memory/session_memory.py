"""
Redis-backed sliding window session memory (last 6 turns).
Falls back to in-memory dict if Redis is unavailable.
"""

import json
import logging
from typing import List
from config import REDIS_URL

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger("cortex.session_memory")


class SessionMemory:
    """
    Sliding-window session store.
    Backend: Redis (preferred) -> in-memory dict (fallback).
    """

    def __init__(self, ttl_hours: int = 24, window_size: int = 6):
        self.ttl = ttl_hours * 3600
        self.window = window_size
        self._store: dict = {}  # in-memory fallback

        try:
            import redis

            self.r = redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            self.r.ping()
            self.backend = "redis"
            logger.info("[SessionMemory] Redis backend active")
        except Exception as e:
            self.r = None
            self.backend = "memory"
            logger.warning(
                "[SessionMemory] Redis unavailable (%s) : using In-Memory fallback "
                "(sessions won't survive restart)",
                e,
            )

    def _key(self, user_id: str) -> str:
        return f"session:{user_id}"

    def save(self, user_id: str, messages: List[BaseMessage]) -> None:
        """Persist the last `window_size` messages for the user."""
        windowed = messages[-self.window :]
        data = [{"type": m.type, "content": m.content} for m in windowed]
        payload = json.dumps(data)

        if self.backend == "redis":
            try:
                self.r.setex(self._key(user_id), self.ttl, payload)
                return
            except Exception as e:
                logger.warning("[SessionMemory] Redis write failed: %s", e)

        self._store[self._key(user_id)] = data

    def load(self, user_id: str) -> List[BaseMessage]:
        """Load stored messages for the user. Returns [] if none found."""
        raw = None

        if self.backend == "redis":
            try:
                raw = self.r.get(self._key(user_id))
            except Exception as e:
                logger.warning("[SessionMemory] Redis read failed: %s", e)

        if raw is None:
            stored = self._store.get(self._key(user_id), [])
            data = stored
        else:
            data = json.loads(raw)

        messages = []
        for d in data:
            if d["type"] == "human":
                messages.append(HumanMessage(content=d["content"]))
            else:
                messages.append(AIMessage(content=d["content"]))
        return messages

    def clear(self, user_id: str) -> None:
        """Remove all session data for a user."""
        if self.backend == "redis":
            try:
                self.r.delete(self._key(user_id))
            except Exception:
                pass
        self._store.pop(self._key(user_id), None)

    def append_and_save(
        self, user_id: str, human_msg: str, ai_msg: str
    ) -> List[BaseMessage]:
        """Load existing history, append new turn, save, and return full window."""
        history = self.load(user_id)
        history.append(HumanMessage(content=human_msg))
        history.append(AIMessage(content=ai_msg))
        self.save(user_id, history)
        return history
