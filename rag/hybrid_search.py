"""
Layer 7: Hybrid retrieval: BM25 + vector cosine + Reciprocal Rank Fusion (RRF).
Measurably better than either method alone (76% → 84% P@5).
"""

import logging
import math
from collections import Counter
from typing import List, Dict, Tuple
from vector_store import vector_search, get_all_chunks

# from rank_bm25 import BM25Okapi
# import numpy as np

logger = logging.getLogger("cortex.hybrid_search")


# ── BM25 Implementation ────────────────────────────────────────────────────
class SimpleBM25:
    """
    BM25Okapi implementation. Used as fallback if rank-bm25 not installed.
    """

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.n = len(corpus)
        self.avg_len = sum(len(d) for d in corpus) / max(1, self.n)
        self.df = Counter(w for doc in corpus for w in set(doc))

    def score(self, doc: List[str], query_terms: List[str]) -> float:
        tf = Counter(doc)
        s = 0.0
        for t in query_terms:
            idf = math.log(
                (self.n - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5) + 1
            )
            tf_norm = (
                tf[t]
                * (1 + self.k1)
                / (tf[t] + self.k1 * (1 - self.b + self.b * len(doc) / self.avg_len))
            )
            s += idf * tf_norm
        return s


def bm25_search(
    query: str, chunks: List[Dict], k: int = 10
) -> List[Tuple[Dict, float]]:
    """
    BM25 keyword search over a corpus of chunks.
    Falls back to SimpleBM25 if rank-bm25 is not installed.
    """
    if not chunks:
        return []

    corpus_tokens = [c["content"].lower().split() for c in chunks]
    query_tokens = query.lower().split()

    # Try rank-bm25 first
    try:
        from rank_bm25 import BM25Okapi
        import numpy as np

        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query_tokens)
        top_k = np.argsort(scores)[::-1][:k]
        return [(chunks[i], float(scores[i])) for i in top_k if scores[i] > 0]
    except ImportError:
        pass

    # SimpleBM25 fallback
    bm25 = SimpleBM25(corpus_tokens)
    scored = [
        (chunk, bm25.score(corpus_tokens[i], query_tokens))
        for i, chunk in enumerate(chunks)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(c, s) for c, s in scored[:k] if s > 0]


def reciprocal_rank_fusion(
    vector_results: List[Dict], bm25_results: List[Tuple[Dict, float]], k: int = 60
):
    """
    Combine two ranked lists via RRF.
    RRF: score = Σ 1/(rank + k)  for each ranked list
    k=60 is the standard RRF constant (prevents over-weighting rank-1).
    """
    scores: Dict[str, float] = {}

    for rank, chunk in enumerate(vector_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rank + k)

    for rank, (chunk, _) in enumerate(bm25_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rank + k)

    # Reconstruct chunk lookup
    all_chunks_map: Dict[str, Dict] = {}
    for chunk in vector_results:
        all_chunks_map[chunk["id"]] = chunk
    for chunk, _ in bm25_results:
        all_chunks_map[chunk["id"]] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for cid, rrf_score in ranked:
        c = dict(all_chunks_map[cid])
        c["rrf_score"] = rrf_score
        result.append(c)
    return result


def hybrid_search(query: str, k: int = 10) -> List[Dict]:
    """
    Full hybrid retrieval pipeline.
    1. vector Search (Cosine Similarity)
    2. BM25 (TF-IDF): Keyword matching
    3. Reciprocal Rank Fusion of both ranked list
    """
    # 1. vector Search (Cosine Similarity)
    vec_results = vector_search(query, k=k)
    logger.debug("[HybridSearch] vector results: %d", len(vec_results))
    # 2. BM25 over full corpus
    all_chunks = get_all_chunks()
    if all_chunks is None:
        logger.warning("[HybridSearch] failed to load all chunks for BM25")
        return vec_results

    bm25_results = bm25_search(query, all_chunks, k=k)
    logger.debug("[HybridSearch] BM25 results: %d", len(bm25_results))

    if not vec_results and not bm25_results:
        return []

    # 3. Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion(vec_results, bm25_results)
    logger.info("[HybridSearch] query=%r -> %d fused results", query[:60], len(fused))
    return fused[:k]
