"""
Layer 4 — PGVector storage with in-memory fallback for offline testing.
Handles CRUD operations on document_chunks table.
"""

import logging
import uuid
from typing import List, Dict
from config import POSTGRES_URL
from embeddings import get_embedding
import psycopg2
from psycopg2.extras import Json

logger = logging.getLogger("cortex.vector_store")
_memory_store: List[Dict] = []


def insert_chunk(
    content: str,
    doc_id: str,
    metadata: Dict,
    access_tier: str,
) -> None:
    """Insert a single chunk with its embedding into PGVector (or memory fallback)"""

    embeddings = get_embedding(content)
    chunk = {
        "id": str(uuid.uuid4()),
        "doc_id": doc_id,
        "content": content,
        "embedding": embeddings,
        "metadata": metadata,
        "access_tier": access_tier,
    }

    pg_url = POSTGRES_URL
    if (
        pg_url
        and not pg_url.startswith("postgresql://cortex:cortex_dev@localhost")
        or (
            pg_url
            and _try_pg_insert(
                chunk, embeddings, content, doc_id, metadata, access_tier
            )
        )
    ):
        return

    _memory_store.append(chunk)


def _try_pg_insert(chunk, embedding, content, doc_id, metadata, access_tier) -> bool:
    try:

        with psycopg2.connect(os.getenv("POSTGRES_URL")) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO document_chunks (doc_id, content, embedding, metadata, access_tier)
                    VALUES (%s, %s, %s::vector, %s, %s)
                    """,
                    (doc_id, content, str(embedding), Json(metadata), access_tier),
                )
        return True
    except Exception as e:
        logger.debug("[VectorStore] PG insert failed: %s", e)
        _memory_store.append(chunk)
        return False
