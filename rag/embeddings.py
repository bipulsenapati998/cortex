"""
Layer 3 — Embedding generation using OpenAI text-embedding-3-small.
Falls back to a simple hash-based mock if no API key is set (for testing).
"""

import logging
import hashlib
from typing import List
from config import EMBED_MODEL, OPEN_API_KEY
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger("cortex.embeddings")

_embed_client = None


def _get_client():
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPEN_API_KEY)
    return _embed_client


def get_embedding(text: str) -> List[float]:
    """
    Generate a 1536-dim embedding vector for the given text.
    Falls back to a deterministic mock if no API key is available.
    """
    api_key = OPEN_API_KEY
    if not api_key or api_key.startswith("sk-proj-"):
        logger.warning("No valid OpenAI API key found. Using mock embedding.")
        return [_mock_embedding(t) for t in text]

    try:
        client = _get_client()
        return client.embed_documents(text)
    except Exception as e:
        logger.warning("[Embeddings] Batch API call failed: %s — using mock", e)
        return [_mock_embedding(t) for t in text]


def _mock_embedding(text: str, dim: int = 1536) -> List[float]:
    """
    Deterministic mock embedding based on character hash values.
    Produces consistent (but not meaningful) vectors for offline testing.
    """

    # Use sha256 seed to generate reproducible floats without struct boundary issues
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    rng_state = seed
    floats = []
    for _ in range(dim):
        rng_state = (
            rng_state * 6364136223846793005 + 1442695040888963407
        ) & 0xFFFFFFFFFFFFFFFF
        val = (rng_state & 0xFFFF) / 0xFFFF * 2 - 1  # range [-1, 1]
        floats.append(val)
    magnitude = sum(x**2 for x in floats) ** 0.5 or 1.0
    return [x / magnitude for x in floats]
