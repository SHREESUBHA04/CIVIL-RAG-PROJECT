from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────
# ANSWER GENERATION PROMPT
# ─────────────────────────────────────────

ANSWER_PROMPT = """You are an expert civil engineering assistant.
Answer the user's question using ONLY the context provided below.

STRICT RULES:
- Base your answer only on the provided context
- If the context does not contain enough information, say exactly:
  "I could not find sufficient information in the documents to answer this question."
- Always mention which document the information came from
- Be precise and technical — this is for civil engineering professionals
- If numbers or standards are mentioned in context, include them in your answer
- Keep your answer focused and clear

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:"""

OUT_OF_SCOPE_RESPONSE = """This question is outside the scope of civil engineering topics 
I'm trained to answer. I can help you with questions about:
- Concrete technology and mix design
- Construction materials and properties  
- IS codes and standards
- Structural concepts
- Civil engineering datasets and statistics

Please ask a civil engineering related question."""


def format_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a clean context string for the LLM.
    Uses the 'context' field (sentence window) not 'text' (single sentence).

    Each chunk is labeled with its source so the LLM can cite it.
    """
    parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.get("source", "Unknown")
        context = chunk.get("context", chunk.get("text", ""))
        parts.append(f"[Source {i+1}: {source}]\n{context}")

    return "\n\n".join(parts)


def generate_answer(
    query: str,
    chunks: List[Dict],
    conversation_history: List[Dict] = None
) -> Dict:
    """
    Generate an answer from retrieved chunks using the LLM.

    Args:
        query                : user's question
        chunks               : top reranked chunks from retrieval
        conversation_history : list of previous messages for context

    Returns dict with:
        - answer   : the generated answer text
        - sources  : list of source documents used
        - chunks   : the chunks that were used
    """
    if not chunks:
        return {
            "answer": "I could not find relevant information in the documents.",
            "sources": [],
            "chunks": []
        }

    # Format context from chunks
    context = format_context(chunks)

    # Build messages — include conversation history for memory
    messages = []

    # Add conversation history (last 6 exchanges = 12 messages)
    if conversation_history:
        messages.extend(conversation_history[-12:])

    # Add current query with context
    messages.append({
        "role": "user",
        "content": ANSWER_PROMPT.format(context=context, query=query)
    })

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # large model for final answer
            messages=messages,
            temperature=0.1,    # low = factual and consistent
            max_tokens=800,
        )

        answer = response.choices[0].message.content.strip()

        # Extract unique sources
        sources = list(set(c["source"] for c in chunks))

        return {
            "answer": answer,
            "sources": sources,
            "chunks": chunks
        }

    except Exception as e:
        return {
            "answer": f"Answer generation failed: {str(e)}",
            "sources": [],
            "chunks": []
        }


def handle_out_of_scope() -> Dict:
    """Return a polite out-of-scope response"""
    return {
        "answer": OUT_OF_SCOPE_RESPONSE,
        "sources": [],
        "chunks": []
    }


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from civil_rag.vectorstore import load_index, search
    from civil_rag.reranker import rerank
    from civil_rag.query_expander import expand_and_search
    from civil_rag.router import route_query

    index, chunks = load_index()

    test_queries = [
        "what is water cement ratio and why is it important",
        "what is the capital of France",
    ]

    for query in test_queries:
        print(f"\n{'═'*60}")
        print(f"QUERY: {query}")
        print('═'*60)

        # Step 1: Route
        category = route_query(query)
        print(f"Route: {category}")

        if category == "OUT_OF_SCOPE":
            result = handle_out_of_scope()
            print(f"\nANSWER:\n{result['answer']}")
            continue

        # Step 2: Expand + Search
        candidates = expand_and_search(query, index, chunks, search, top_k=20)

        # Step 3: Rerank
        final_chunks = rerank(query, candidates, top_k=5)

        # Step 4: Generate
        result = generate_answer(query, final_chunks)

        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES: {result['sources']}")