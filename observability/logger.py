import json
import logging
from datetime import datetime
from time import time
from config import LOG_LEVEL
from dataclasses import dataclass, asdict
from typing import Optional

# Configure root logger
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cortex")


@dataclass
class QueryLog:
    user_id: str
    query: str
    agent_used: str
    tokens: int
    cost_usd: float
    latency_ms: int
    p_at_5: Optional[float] = None
    intent: Optional[str] = None
    error: Optional[str] = None


def log_query(
    user_id: str,
    query: str,
    agent_used: str,
    tokens: int,
    cost_usd: float,
    latency_ms: int,
    p_at_5: Optional[float] = None,
    intent: Optional[str] = None,
    error: Optional[str] = None,
):
    """Log a completed query with full observability metadata."""
    record = QueryLog(
        user_id=user_id,
        query=query[:200],  # truncate for log safety
        agent_used=agent_used,
        tokens=tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        p_at_5=p_at_5,
        intent=intent,
        error=error,
    )
    logger.info("QUERY | %s", json.dumps(asdict(record)))

        # Optionally persist to PostgreSQL if available
    _persist_to_db(record)


def _persist_to_db(record: QueryLog):
    """Best-effort write to query_log table — never blocks the response."""
    try:
        import psycopg2

        postgres_url = os.getenv("POSTGRES_URL")
        if not postgres_url:
            return
        with psycopg2.connect(postgres_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_log
                        (user_id, query, agent_used, tokens, cost_usd, latency_ms, p_at_5)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record.user_id,
                        record.query,
                        record.agent_used,
                        record.tokens,
                        record.cost_usd,
                        record.latency_ms,
                        record.p_at_5,
                    ),
                )
    except Exception:
        pass  # observability must never break the main flow


def log_retrieval(
    raw_query: str,
    expanded_query: str,
    num_results: int,
    user_tier: str,
):
    """Log a RAG retrieval event."""
    logger.info(
        "RETRIEVAL | tier=%s results=%d raw=%r expanded=%r",
        user_tier,
        num_results,
        raw_query[:100],
        expanded_query[:100],
    )

class Timer:
    """Context manager for measuring latency."""

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = int((time.time() - self._start) * 1000)