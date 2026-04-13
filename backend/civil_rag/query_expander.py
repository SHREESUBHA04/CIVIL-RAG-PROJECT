from typing import List
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# ─────────────────────────────────────────
# CLIENT
# ─────────────────────────────────────────

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────
# WHY QUERY EXPANSION?
# ─────────────────────────────────────────
#
# User asks: "what is w/c ratio?"
#
# Problem: FAISS looks for vectors similar to "what is w/c ratio"
# But your document says "water-cement ratio" not "w/c ratio"
# These embed slightly differently → good chunks get missed
#
# Solution: expand the query into 3 versions before searching
#
# Version 1: "what is w/c ratio?"                     ← original
# Version 2: "water cement ratio definition concrete"  ← rephrased
# Version 3: "effect of water cement ratio on strength"← related angle
#
# Search all 3 → merge results → rerank
# Much higher chance of finding the right chunk
#
# ─────────────────────────────────────────

EXPANSION_PROMPT = """You are a civil engineering expert helping improve document search.

Given a user's question, generate exactly 3 search queries.

STRICT RULES:
- Output ONLY the queries, nothing else
- No numbering like "Query 1:" or "1."
- No labels, no explanations, no blank lines
- One query per line, exactly 3 lines total

Query 1: Rephrase using different technical terms or expanded abbreviations
Query 2: Focus on the core concept only
Query 3: Related practical application

User question: {query}"""


def expand_query(query: str) -> List[str]:
    """
    Takes a user query and returns 4 search queries:
    - The original query
    - 3 LLM-generated variations

    All 4 are used for FAISS search.
    Results are merged and deduplicated before reranking.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # use small fast model for expansion
            messages=[
                {
                    "role": "user",
                    "content": EXPANSION_PROMPT.format(query=query)
                }
            ],
            temperature=0.3,    # low temperature = focused, consistent output
            max_tokens=150,
        )

        raw = response.choices[0].message.content.strip()

        # Parse the 3 lines
        expanded = [line.strip() for line in raw.split("\n") if line.strip()]

        # Keep only first 3 in case model returned more
        expanded = expanded[:3]

        # Always include original query first
        all_queries = [query] + expanded

        print(f"\nQuery expansion:")
        for i, q in enumerate(all_queries):
            label = "original" if i == 0 else f"expanded {i}"
            print(f"  [{label}] {q}")

        return all_queries

    except Exception as e:
        print(f"Query expansion failed: {e}")
        # Fallback — just use original query
        return [query]


def expand_and_search(
    query: str,
    index,
    chunks,
    search_fn,
    top_k: int = 20
) -> List[dict]:
    """
    Expand query → search all variations → merge results.

    Deduplication: if the same chunk appears in multiple
    search results, keep only the highest scoring version.

    Args:
        query     : original user question
        index     : FAISS index
        chunks    : chunk list
        search_fn : the search() function from vectorstore.py
        top_k     : candidates per query before merging
    """
    queries = expand_query(query)

    seen_ids = {}   # chunk_id → best result so far

    for q in queries:
        results = search_fn(q, index, chunks, top_k=top_k)
        for result in results:
            cid = result["chunk_id"]
            # Keep the version with the highest FAISS score
            if cid not in seen_ids or result["score"] > seen_ids[cid]["score"]:
                seen_ids[cid] = result

    # Merge and sort by FAISS score
    merged = sorted(seen_ids.values(), key=lambda x: x["score"], reverse=True)

    print(f"\n  {len(queries)} queries × {top_k} results = "
          f"{len(queries)*top_k} raw → {len(merged)} unique after dedup")

    # Return top_k for reranker
    return merged[:top_k]


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from civil_rag.vectorstore import load_index, search
    from civil_rag.reranker import rerank

    index, chunks = load_index()

    query = "what is w/c ratio"
    print(f"Original query: '{query}'")

    # Expand + search
    candidates = expand_and_search(query, index, chunks, search, top_k=20)

    # Rerank
    final = rerank(query, candidates, top_k=5)

    print(f"\n--- Final Top 5 After Expansion + Reranking ---")
    for i, c in enumerate(final):
        print(f"\n[{i+1}] Rerank score: {c['rerank_score']:.4f}")
        print(f"     Text: {c['text'][:100]}...")
        print(f"     Source: {c['source']}")