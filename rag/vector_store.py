"""
Layer 4 — PGVector storage with in-memory fallback for offline testing.
Handles CRUD operations on document_chunks table.
"""

import os
import logging
import uuid
from typing import List, Dict


logger = logging.getLogger("cortex.vector_store")


def insert_chunk(
    content: str,
    doc_id: str,
    metadata: Dict,
    access_tier: str,
    chunk_id: str = None,
) -> None:
    """Insert a single chunk with its embedding into PGVector (or memory fallback)"""

    embeddings = get_embedding(content)
