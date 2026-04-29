import json
import logging
from datetime import datetime
from config import LOG_LEVEL

logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cortex")


def log_query(
    user_id: str, query: str, agent: str, token: int, latency_ms: float, cost_usd: float
):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "query": query,
        "agent": agent,
        "token_count": token,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
    }
    logger.info(json.dumps(log_entry))
