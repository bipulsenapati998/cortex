import logging
from typing import Optional, List, Dict
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from config import LLM_MODEL
from vector_store import insert_chunk
from query_understanding import understand_query

from ingestion.document_loader import load_text_file, paragraph_chunk, _infer_metadata


logger = logging.getLogger("cortex.rag_pipeline")
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return _llm


# ── Ingestion ─────────────────────────────────────────────────────────────
def ingest_document(folder_path: str) -> int:
    """
    Ingest all .txt & .md files in a folder into the vector store
    Returns the number of chunks ingested
    """
    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"[Ingest] Folder '{folder_path}' does not exist")
        return 0
    total_chunks = 0
    for filepath in folder.iterdir():
        if filepath.suffix.lower() not in {".txt", ".md"}:
            continue

        logger.info("[Ingest] Processing: %s", filepath.name)

        try:
            text = load_text_file(str(filepath))
            meta = _infer_metadata(filepath.name)
            chunks = paragraph_chunk(text)

            for i, chunk_text in enumerate(chunks):
                insert_chunk(
                    content=chunk_text,
                    doc_id=meta["doc_id"],
                    metadata={**meta, "chunk_index": i},
                    access_tier=meta["access_tier"],
                )
            total_chunks += len(chunks)
            logger.info(
                "[Ingest] %s → %d chunks (tier=%s)",
                filepath.name,
                len(chunks),
                meta["access_tier"],
            )
        except Exception as e:
            logger.error("[Ingest] Failed on %s: %s", filepath.name, e)

    logger.info("[Ingest] Completed: %d total chunks", total_chunks)
    return total_chunks


# ── Query pipeline ─────────────────────────────────────────────────────────
def query(
    raw_query: str,
    user_tier: str = "standard",
    k: int = 5,
    entity_context: Optional[str] = None,
    session_history: Optional[List] = None,
) -> Dict:
    """
    Run the full 7-layer RAG pipeline.

    Returns:
        {
            "answer":         str,
            "sources":        list of chunk dicts,
            "expanded_query": str,
            "num_results":    int,
        }
    """
    # Layer 5: Query understanding + expansion
    understood = understand_query(raw_query)
    expanded_query = understood["expanded"]

    # Layer 7: Hybrid search (calls embeddings L3 + PGVector L4 + BM25)
    raw_results = hybrid_search(expanded_query, k=k * 2)  # over-fetch then filter

    # Layer 6: Access control — remove docs user is not allowed to see
    filtered = filter_by_tier(raw_results, user_tier)
    results = filtered[:k]

    log_retrieval(raw_query, expanded_query, len(results), user_tier)

    if not results:
        return {
            "answer": (
                "I couldn't find relevant information in the knowledge base for your query. "
                "Please try rephrasing, or contact HR/IT directly if this is urgent."
            ),
            "sources": [],
            "expanded_query": expanded_query,
            "num_results": 0,
        }

    # Generate grounded answer
    answer = generate_answer(
        raw_query,
        results,
        entity_context=entity_context,
        session_history=session_history,
    )

    return {
        "answer": answer,
        "sources": results,
        "expanded_query": expanded_query,
        "num_results": len(results),
    }
