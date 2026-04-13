import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Tuple
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────
# WHY A REFLECTION AGENT?
# ─────────────────────────────────────────
#
# Without reflection, the pipeline is a straight line:
#   Query → Retrieve → Generate → Done
#
# Problems with straight line:
#   - If retrieval found wrong chunks, answer is wrong
#   - System has no way to know it gave a bad answer
#   - No second chance
#
# With reflection, the pipeline can loop:
#   Query → Retrieve → Generate → Reflect
#                                    ↓ bad answer?
#                              Refine query → Retrieve again
#                                    ↓ good answer?
#                              Return to user
#
# This is a real agentic loop — not a fake confidence score.
# The agent actually DOES something when quality is low.
#
# We limit to 1 retry to avoid infinite loops.
# In production systems (like AutoGPT) this can be N retries.
#
# ─────────────────────────────────────────

REFLECTION_PROMPT = """You are a quality checker for a civil engineering AI assistant.

Evaluate whether this answer properly addresses the question.

QUESTION: {query}

ANSWER: {answer}

RETRIEVED CONTEXT USED:
{context_preview}

Evaluate on these criteria:
1. RELEVANCE: Does the answer actually address what was asked?
2. GROUNDED: Is the answer based on the retrieved context (not hallucinated)?
3. COMPLETE: Does it provide enough information to be useful?
4. SPECIFIC: Does it include specific values, standards, or technical details where available?

Respond in this EXACT format (no extra text):
PASS_OR_FAIL: PASS or FAIL
REASON: one sentence explaining why
REFINED_QUERY: if FAIL, write a better search query to find the missing information. If PASS, write NONE"""


def reflect(
    query: str,
    answer: str,
    chunks: List[Dict],
) -> Tuple[bool, str, str]:
    """
    Reflect on the quality of a generated answer.

    Args:
        query  : original user question
        answer : generated answer to evaluate
        chunks : chunks that were used to generate the answer

    Returns:
        passed        : True if answer is good, False if needs retry
        reason        : explanation of the decision
        refined_query : better query to use on retry (if failed)
    """
    # Build a preview of the context that was used
    context_preview = "\n".join([
        f"- {c['text'][:100]}..." for c in chunks[:3]
    ])

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # fast model for reflection
            messages=[{
                "role": "user",
                "content": REFLECTION_PROMPT.format(
                    query=query,
                    answer=answer,
                    context_preview=context_preview
                )
            }],
            temperature=0.0,
            max_tokens=150,
        )

        raw = response.choices[0].message.content.strip()

        # Parse the structured response
        lines = {}
        for line in raw.split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                lines[key.strip()] = value.strip()

        passed = lines.get("PASS_OR_FAIL", "PASS").upper() == "PASS"
        reason = lines.get("REASON", "No reason provided")
        refined_query = lines.get("REFINED_QUERY", "NONE")

        # Clean up refined query
        if refined_query.upper() == "NONE" or not refined_query:
            refined_query = None

        return passed, reason, refined_query

    except Exception as e:
        print(f"  Reflection failed: {e}, defaulting to PASS")
        return True, "Reflection error — defaulting to pass", None


# ─────────────────────────────────────────
# FULL AGENTIC PIPELINE WITH REFLECTION
# ─────────────────────────────────────────

def run_with_reflection(
    query: str,
    index,
    chunks_store: List[Dict],
    embedder,
    route_vectors: Dict,
    route_fn,
    search_fn,
    expand_fn,
    rerank_fn,
    generate_fn,
    handle_oos_fn,
    conversation_history: List[Dict] = None,
) -> Dict:
    """
    Full pipeline with agentic reflection loop.

    Flow:
    1. Route query
    2. Expand + Search + Rerank
    3. Generate answer
    4. Reflect on answer quality
    5. If failed → retry once with refined query
    6. Return best answer

    Args are the functions from other modules —
    this keeps reflection.py decoupled from the others.
    """

    # ── Step 1: Route ────────────────────────────────────────────
    category = route_fn(query, embedder, route_vectors, verbose=True)

    if category == "OUT_OF_SCOPE":
        return handle_oos_fn()

    if category == "DATA":
        # Data pipeline placeholder — returns honest message for now
        # Will be replaced by dataset_analyzer in next module
        return {
            "answer": "Data analysis pipeline — coming in next module.",
            "sources": [],
            "chunks": [],
            "reflection": {"passed": True, "reason": "Data route", "retried": False}
        }

    # ── Step 2: Expand + Search + Rerank ─────────────────────────
    print("\n[Step 2] Retrieving relevant chunks...")
    candidates = expand_fn(query, index, chunks_store, search_fn, top_k=20)
    final_chunks = rerank_fn(query, candidates, top_k=5)

    if not final_chunks:
        return {
            "answer": "I could not find relevant information in the documents.",
            "sources": [],
            "chunks": [],
            "reflection": {"passed": False, "reason": "No chunks retrieved", "retried": False}
        }

    # ── Step 3: Generate first answer ────────────────────────────
    print("\n[Step 3] Generating answer...")
    result = generate_fn(query, final_chunks, conversation_history)
    answer = result["answer"]

    # ── Step 4: Reflect ──────────────────────────────────────────
    print("\n[Step 4] Reflecting on answer quality...")
    passed, reason, refined_query = reflect(query, answer, final_chunks)

    print(f"  Reflection: {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"  Reason: {reason}")

    if passed:
        result["reflection"] = {
            "passed": True,
            "reason": reason,
            "retried": False
        }
        return result

    # ── Step 5: Retry with refined query (once only) ─────────────
    print(f"\n[Step 5] Retrying with refined query: '{refined_query}'")

    retry_candidates = expand_fn(
        refined_query, index, chunks_store, search_fn, top_k=20
    )
    retry_chunks = rerank_fn(refined_query, retry_candidates, top_k=5)
    retry_result = generate_fn(query, retry_chunks, conversation_history)

    # Reflect once more on retry — don't loop again
    retry_passed, retry_reason, _ = reflect(
        query, retry_result["answer"], retry_chunks
    )
    print(f"  Retry reflection: {'PASS ✓' if retry_passed else 'FAIL ✗'}")
    print(f"  Reason: {retry_reason}")

    retry_result["reflection"] = {
        "passed": retry_passed,
        "reason": retry_reason,
        "retried": True,
        "refined_query": refined_query,
        "original_answer": answer,
    }

    return retry_result


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    from civil_rag.vectorstore import load_index, search, embedder
    from civil_rag.reranker import rerank
    from civil_rag.query_expander import expand_and_search
    from civil_rag.router import build_route_embeddings, route_query
    from civil_rag.generator import generate_answer, handle_out_of_scope

    # Load everything
    index, chunks_store = load_index()
    route_vectors = build_route_embeddings(embedder)

    test_queries = [
        "what is water cement ratio and why does it matter",
        "what is the capital of France",
    ]

    for query in test_queries:
        print(f"\n{'═'*60}")
        print(f"QUERY: {query}")
        print('═'*60)

        result = run_with_reflection(
            query=query,
            index=index,
            chunks_store=chunks_store,
            embedder=embedder,
            route_vectors=route_vectors,
            route_fn=route_query,
            search_fn=search,
            expand_fn=expand_and_search,
            rerank_fn=rerank,
            generate_fn=generate_answer,
            handle_oos_fn=handle_out_of_scope,
        )

        print(f"\n{'─'*60}")
        print(f"FINAL ANSWER:\n{result['answer']}")
        print(f"\nSOURCES: {result.get('sources', [])}")
        reflection = result.get("reflection", {})
        print(f"REFLECTION: {'PASS' if reflection.get('passed') else 'FAIL'}")
        print(f"RETRIED: {reflection.get('retried', False)}")