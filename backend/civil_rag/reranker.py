from typing import List, Dict
from sentence_transformers import CrossEncoder

# ─────────────────────────────────────────
# LOAD RERANKER MODEL (once, globally)
# ─────────────────────────────────────────

# This model takes (query, passage) pairs and scores
# how relevant the passage is to the query.
# Much more accurate than embedding similarity alone
# because it looks at BOTH texts together, not separately.

print("Loading reranker model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Reranker ready.")


# ─────────────────────────────────────────
# WHY A RERANKER?
# ─────────────────────────────────────────
#
# FAISS embedding search:
#   Query → embed → find nearest vectors
#   Problem: embeds query and chunks SEPARATELY
#   "water cement ratio" and "w/c ratio" might not be
#   close in embedding space even though they mean the same thing
#
# Cross-encoder reranker:
#   Takes (query, chunk) as ONE input
#   Reads both together and scores relevance
#   Much more accurate but too slow to run on all chunks
#
# Solution: FAISS fetches top 20 candidates fast,
#           reranker picks best 5 from those 20 precisely.
#           Best of both worlds.
#
# ─────────────────────────────────────────


def rerank(
    query: str,
    candidates: List[Dict],
    top_k: int = 5,
    min_score: float = 0.0   # ← add this parameter
) -> List[Dict]:
    """
    Rerank FAISS candidates using cross-encoder.
    Filters out results with negative relevance scores.
    """
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    # Sort by score
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    # Filter out negative scores — these are irrelevant chunks
    # The cross-encoder uses 0 as the relevance boundary
    reranked = [r for r in reranked if r["rerank_score"] >= min_score]

    if not reranked:
        # Safety fallback — if everything scored negative,
        # return top 2 anyway rather than empty
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:2]

    return reranked[:top_k]

# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from civil_rag.vectorstore import load_index, search

    # Load index
    index, chunks = load_index()

    # Search
    query = "what is water cement ratio"
    print(f"Query: '{query}'")

    candidates = search(query, index, chunks, top_k=20)
    print(f"\nBefore reranking — Top 5 FAISS results:")
    for i, c in enumerate(candidates[:5]):
        print(f"  [{i+1}] FAISS score: {c['score']:.4f} | {c['text'][:80]}...")

    # Rerank
    reranked = rerank(query, candidates, top_k=5)
    print(f"\nAfter reranking — Top 5 reranked results:")
    for i, c in enumerate(reranked):
        print(f"  [{i+1}] Rerank score: {c['rerank_score']:.4f} | {c['text'][:80]}...")

    print("\n--- Best Result Context (what LLM will read) ---")
    print(reranked[0]["context"])
    print(f"\nSource: {reranked[0]['source']}")