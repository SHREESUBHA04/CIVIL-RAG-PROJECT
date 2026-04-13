import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────
# ROUTE DEFINITIONS
# ─────────────────────────────────────────
# Each category has example sentences that represent it.
# More examples = better coverage = more accurate routing.
# These are embedded ONCE at startup, never again.

ROUTE_EXAMPLES = {
    "DOCUMENT": [
        # Concept questions
        "what is water cement ratio",
        "explain concrete mix design",
        "what are the types of cement",
        "how does curing affect concrete strength",
        "what is workability of concrete",
        "explain hydration of cement",
        "what is slump test",
        "types of foundations in construction",
        "explain prestressed concrete",
        "what causes concrete failure",
        # IS code specific — always DOCUMENT
        "IS 456 provisions for reinforcement cover",
        "what does IS 10262 say about mix design",
        "IS code recommendation for water cement ratio",
        "what is the IS code limit for cement content",
        # Grade specific — always DOCUMENT
        "cement content for M25 concrete",
        "water cement ratio for M20 grade",
        "what is mix proportion for M30",
        "minimum cement content for M15 as per IS code",
        "what is characteristic strength of M25",
        # Table/figure questions from documents
        "what does the table show for cement ratio",
        "what are the values in the mix design table",
        "what does figure 3 show about strength",
        "what is the value from the stress strain curve",
        # Physics used in civil engineering
        "what is work energy theorem in structures",
        "explain strain energy in beams",
        "what is virtual work principle",
        "force displacement relationship in concrete",
        "what is the excavation quantity in the tender",
        "how much does excavation cost",
        "what is the amount for concrete work",
        "what is the scope of work",
        "what are the items in the bill of quantities",
        "what is the rate for steel reinforcement",
        "what is the total project cost"
    ],

    "DATA": [
        # Must explicitly reference dataset/CSV/records
        "what is the average compressive strength in the dataset",
        "show me the maximum value in the CSV",
        "what is the correlation between variables in the data",
        "how many samples are in the dataset",
        "standard deviation of strength values in our records",
        "which record has the highest cement content",
        "statistical summary of the concrete dataset",
        "what is the mean water content across all samples",
        "plot the relationship between variables in the data",
        "what percentage of samples exceed 30 MPa in the dataset",
        "show distribution of values in our data",
        "compare all entries in the dataset",
        "minimum value recorded in our CSV file",
    ],

    "OUT_OF_SCOPE": [
        "what is the capital of France",
        "who won the cricket world cup",
        "what is the weather today",
        "tell me a joke",
        "how do I cook pasta",
        "what is the stock price of Apple",
        "explain quantum physics",
        "what is machine learning",
        "who is the prime minister",
        "how do I fix my computer",
    ]
}
# Confidence threshold — below this, fall back to LLM
# 0.5 means: if best match is less than 50% similar, use LLM
CONFIDENCE_THRESHOLD = 0.50


# ─────────────────────────────────────────
# BUILD ROUTE EMBEDDINGS AT STARTUP
# ─────────────────────────────────────────

def build_route_embeddings(
    embedder: SentenceTransformer
) -> Dict[str, np.ndarray]:
    """
    Embed all example sentences for each category.
    Average them into a single representative vector per category.

    This is called ONCE when the app starts.
    Result is stored in memory — no re-computation needed.

    Why average?
    Each example sentence embeds to a point in 384-dim space.
    The average of all examples is the "center" of that category.
    Queries close to this center belong to this category.
    """
    print("Building semantic route embeddings...")
    route_vectors = {}

    for category, examples in ROUTE_EXAMPLES.items():
        # Embed all examples for this category
        embeddings = embedder.encode(
            examples,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # Average into one representative vector
        centroid = embeddings.mean(axis=0)
        # Re-normalize the centroid
        centroid = centroid / np.linalg.norm(centroid)
        route_vectors[category] = centroid
        print(f"  {category}: {len(examples)} examples → 1 centroid vector")

    print("Route embeddings ready.\n")
    return route_vectors


def semantic_route(
    query: str,
    embedder: SentenceTransformer,
    route_vectors: Dict[str, np.ndarray],
) -> Tuple[str, float, Dict[str, float]]:
    """
    Route a query using embedding similarity.

    Returns:
        category   : best matching category
        confidence : similarity score of best match (0-1)
        all_scores : similarity scores for all categories
    """
    # Embed the query
    query_vec = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    # Cosine similarity against each category centroid
    # Since both are normalized, dot product = cosine similarity
    scores = {}
    for category, centroid in route_vectors.items():
        scores[category] = float(np.dot(query_vec, centroid))

    # Pick best category
    best_category = max(scores, key=scores.get)
    confidence = scores[best_category]

    return best_category, confidence, scores


# ─────────────────────────────────────────
# LLM FALLBACK (only for ambiguous queries)
# ─────────────────────────────────────────

FALLBACK_PROMPT = """Classify this question into exactly one category:
- DOCUMENT: civil engineering concepts, materials, standards, methods
- DATA: statistics, numbers, averages from datasets
- OUT_OF_SCOPE: completely unrelated to civil engineering

Reply with ONLY one word: DOCUMENT, DATA, or OUT_OF_SCOPE

Question: {query}"""


def llm_fallback_route(query: str) -> str:
    """
    LLM-based routing — only called when semantic routing
    confidence is below threshold.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": FALLBACK_PROMPT.format(query=query)
            }],
            temperature=0.0,
            max_tokens=10,
        )
        result = response.choices[0].message.content.strip().upper()
        if result not in {"DOCUMENT", "DATA", "OUT_OF_SCOPE"}:
            return "DOCUMENT"
        return result
    except Exception:
        return "DOCUMENT"


# ─────────────────────────────────────────
# MAIN ROUTER FUNCTION
# ─────────────────────────────────────────
# Keywords that ALWAYS mean DOCUMENT regardless of embeddings
DOCUMENT_KEYWORDS = [
    # existing keywords...
    "is code", "is 456", "is 10262", "is 383",
    "m10", "m15", "m20", "m25", "m30", "m35", "m40",
    "figure", "fig.", "table", "as per", "as per code",
    "work energy", "strain energy", "virtual work",
    "the document", "the pdf", "the book",

    # Construction activity keywords — always DOCUMENT
    "excavation", "footing", "foundation", "reinforcement",
    "shuttering", "formwork", "curing", "compaction",
    "plastering", "brickwork", "masonry", "concreting",
    "pile", "retaining wall", "beam", "column", "slab",
    "bar bending", "bbs", "raft", "plinth",
]

# Keywords that ALWAYS mean DATA
DATA_KEYWORDS = [
    "in the dataset",
    "in the csv",
    "in our data",
    "in the records",
    "across all samples",
    "from the data",
    "in the file",
    "show me the data",
    "statistical",
    "average of all",
    "mean of all",
    # Note: "amount" alone removed — too ambiguous
]

def rule_based_check(query: str):
    """
    Check obvious routing rules before hitting the embedder.
    Returns category string or None if no rule matches.
    """
    q_lower = query.lower()

    for kw in DOCUMENT_KEYWORDS:
        if kw in q_lower:
            return "DOCUMENT"

    for kw in DATA_KEYWORDS:
        if kw in q_lower:
            return "DATA"

    return None  # no rule matched, use semantic routing

def route_query(
    query: str,
    embedder,
    route_vectors: Dict[str, np.ndarray],
    verbose: bool = True
) -> str:
    # Step 0: Rule-based check first (instant, no embedder needed)
    rule_result = rule_based_check(query)
    if rule_result:
        if verbose:
            print(f"\nRouting: '{query}'")
            print(f"  → Rule-based route: {rule_result}")
        return rule_result

    # Step 1: Semantic routing
    category, confidence, all_scores = semantic_route(
        query, embedder, route_vectors
    )

    if verbose:
        print(f"\nRouting: '{query}'")
        for cat, score in sorted(
            all_scores.items(), key=lambda x: -x[1]
        ):
            marker = "← best" if cat == category else ""
            print(f"  {cat:15s}: {score:.4f} {marker}")

    if confidence >= CONFIDENCE_THRESHOLD:
        if verbose:
            print(f"  → Semantic route: {category} "
                  f"(confidence: {confidence:.4f})")
        return category

    if verbose:
        print(f"  → Low confidence ({confidence:.4f}), "
              f"using LLM fallback...")
    llm_result = llm_fallback_route(query)
    if verbose:
        print(f"  → LLM route: {llm_result}")
    return llm_result


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    # Use the same embedder as vectorstore (loaded once, shared)
    from civil_rag.vectorstore import embedder

    # Build route vectors once
    route_vectors = build_route_embeddings(embedder)

    test_queries = [
        "what is water cement ratio",
        "what is the average compressive strength in the dataset",
        "what is the capital of France",
        "explain IS 456 provisions for cover to reinforcement",
        "show me correlation between w/c ratio and strength",
        "who won the cricket world cup",
        "what are the types of cement",
        "maximum value of concrete strength in CSV",
        "how does aggregate size affect workability",
        "tell me a joke",
    ]

    print("\n" + "═"*60)
    print("SEMANTIC ROUTER TEST")
    print("═"*60)

    results = []
    for query in test_queries:
        category = route_query(query, embedder, route_vectors)
        results.append((category, query))

    print("\n" + "─"*60)
    print("SUMMARY:")
    print("─"*60)
    for category, query in results:
        print(f"  [{category:15s}] {query}")