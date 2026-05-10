"""
Layer 4: PGVector storage with in-memory fallback for offline testing.
Handles CRUD operations on document_chunks table.
"""

import os
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


def vector_search(query: str, k: int = 5) -> List[Dict]:
    """
    Retrieve top-k chunks by cosine similarity to the query embedding.
    Uses PGVector if available, otherwise brute-force in-memory search.
    """

    query_vec = get_embedding(query)

    pg_url = POSTGRES_URL
    if pg_url and not pg_url.startswith("postgresql://cortex:cortex_dev@localhost"):
        results = _pg_vector_search(query_vec, k)
        if results is not None:
            return results

    return _memory_vector_search(query_vec, k)


def _pg_vector_search(query_vec: List[float], k: int) -> List[Dict]:
    try:
        with psycopg2.connect(POSTGRES_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id::text, doc_id, content, metadata, access_tier,
                           1 - (embedding <=> %s::vector) AS score
                    FROM document_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (str(query_vec), str(query_vec), k),
                )
                rows = cur.fetchall()
                return [
                    {
                        "id": row[0],
                        "doc_id": row[1],
                        "content": row[2],
                        "metadata": row[3] or {},
                        "access_tier": row[4],
                        "score": float(row[5]),
                    }
                    for row in rows
                ]
    except Exception as e:
        logger.debug("[VectorStore] PG search failed: %s", e)
        return None


def _memory_vector_search(query_vec: List[float], k: int) -> List[Dict]:
    """Brute-force cosine similarity over in-memory chunks."""
    if not _memory_store:
        return []

    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x**2 for x in a) ** 0.5 or 1.0
        mag_b = sum(x**2 for x in b) ** 0.5 or 1.0
        return dot / (mag_a * mag_b)

    scored = [
        (chunk, cosine_sim(query_vec, chunk["embedding"])) for chunk in _memory_store
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    results = []
    for chunk, score in scored[:k]:
        r = dict(chunk)
        r["score"] = score
        results.append(r)
    return results


def get_all_chunks() -> List[Dict]:
    """
    Return all chunks from PGVector or in-memory store for BM25 Corpus building.
    Chunks are dicts with keys: id, doc_id, content, metadata, access_tier, score (0.0).
    """
    pg_url = POSTGRES_URL
    if pg_url and not pg_url.startswith("postgresql://cortex:cortex_dev@localhost"):
        try:
            with psycopg2.connect(pg_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id::text, doc_id, content, metadata, access_tier FROM document_chunks"
                    )
                    return [
                        {
                            "id": row[0],
                            "doc_id": row[1],
                            "content": row[2],
                            "metadata": row[3] or {},
                            "access_tier": row[4],
                        }
                        for row in cur.fetchall()
                    ]
        except Exception as e:
            logger.error("[VectorStore] Failed to fetch all chunks for BM25: %s", e)
            return []
    return [{k: v for k, v in c.items() if k != "embedding"} for c in _memory_store]


def clear_store() -> None:
    """Clear all chunks (used in tests)."""
    global _memory_store
    _memory_store = []
