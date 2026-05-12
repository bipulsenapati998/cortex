import os
from typing import Dict
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POSTGRES_URL = os.getenv(
    "POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/cortex"
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_COST_PER_QUERY = float(os.getenv("MAX_COST_PER_QUERY", "0.01"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

KNOWLEDGE_AGENT_PROMPT_VERSION = os.getenv("KNOWLEDGE_AGENT_PROMPT_VERSION", "v1.0.0")
RESEARCH_AGENT_PROMPT_VERSION = os.getenv("RESEARCH_AGENT_PROMPT_VERSION", "v1.0.0")
SUPERVISOR_PROMPT_VERSION = os.getenv("SUPERVISOR_PROMPT_VERSION", "v1.0.0")

# Maps user_tier -> set of allowed document access_tier values
TIER_PERMISSIONS: Dict[str, set] = {
    "standard": {"public", "internal"},
    "manager": {"public", "internal", "confidential"},
    "exec": {"public", "internal", "confidential", "secret"},
}
