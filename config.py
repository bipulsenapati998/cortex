import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

OPEN_API_KEY= os.getenv("OPENAI_API_KEY")
REDIS_URL= os.getenv("REDIS_URL", "redis://localhost:6379")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/cortex")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_COST_PER_QUERY = float(os.getenv("MAX_COST_PER_QUERY", "0.01"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")