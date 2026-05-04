import json
import logging
from datetime import datetime
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
