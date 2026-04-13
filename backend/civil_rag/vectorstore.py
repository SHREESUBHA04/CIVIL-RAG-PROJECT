import os
import json
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

VECTORSTORE_DIR = Path(__file__).parent.parent / "vectorstore"
VECTORSTORE_DIR.mkdir(exist_ok=True)

INDEX_PATH    = VECTORSTORE_DIR / "index.faiss"
CHUNKS_PATH   = VECTORSTORE_DIR / "chunks.pkl"

# This model is free, runs locally, no API key needed
# 384-dimensional embeddings, very fast
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ─────────────────────────────────────────
# LOAD EMBEDDING MODEL (once, globally)
# ─────────────────────────────────────────

print("Loading embedding model...")
embedder = SentenceTransformer(MODEL_NAME)
print("Embedding model ready.")


# ─────────────────────────────────────────
# BUILD INDEX
# ─────────────────────────────────────────

def build_index(chunks: List[Dict]) -> None:
    """
    Builds a production-grade FAISS index using IVFPQ.

    IVF  = Inverted File Index (clustering for fast search)
    PQ   = Product Quantization (compression for memory efficiency)

    How IVF works:
    - Divides all vectors into 'nlist' clusters (like zip codes)
    - At search time, only checks 'nprobe' nearest clusters
    - nprobe=10 means: check 10 clusters out of nlist
    - Trade-off: higher nprobe = more accurate but slower

    How PQ works:
    - Compresses each 384-dim vector into M subvectors
    - Each subvector quantized to 256 possible values (8 bits)
    - Reduces memory from 1536 bytes to M bytes per vector
    """
    print(f"\nEmbedding {len(chunks)} chunks...")

    sentences = [chunk["text"] for chunk in chunks]

    embeddings = embedder.encode(
        sentences,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    print(f"Embeddings shape: {embeddings.shape}")

    dimension = embeddings.shape[1]  # 384
    n_vectors  = len(embeddings)

    # ── Choose index type based on dataset size ──────────────────
    #
    # Under 1,000 vectors:
    #   FlatIP — exact search, fast enough, no training needed
    #
    # 1,000 – 100,000 vectors:
    #   IVFFlat — clustering + exact search within clusters
    #   nlist = sqrt(n_vectors) is the standard rule of thumb
    #
    # Over 100,000 vectors:
    #   IVFPQ — clustering + compressed search
    #   Saves memory, slightly less accurate
    #
    # ─────────────────────────────────────────────────────────────

    if n_vectors < 1000:
        print("Small dataset — using FlatIP (exact search)")
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

    elif n_vectors < 100_000:
        print("Medium dataset — using IVFFlat")
        nlist = max(4, int(n_vectors ** 0.5))  # sqrt rule
        print(f"  nlist (clusters) = {nlist}")

        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(
            quantizer,
            dimension,
            nlist,
            faiss.METRIC_INNER_PRODUCT
        )

        # IVF must be trained before adding vectors
        # Training finds the cluster centroids
        print("  Training index...")
        index.train(embeddings)
        index.add(embeddings)

        # nprobe = how many clusters to search at query time
        # Higher = more accurate, slower
        # 10-20% of nlist is a good starting point
        index.nprobe = max(1, nlist // 10)
        print(f"  nprobe = {index.nprobe}")

    else:
        print("Large dataset — using IVFPQ")
        nlist = 1024
        M     = 48    # number of subvectors (must divide dimension)
                      # 384 / 48 = 8 — works cleanly
        nbits = 8     # bits per subvector (256 centroids)

        print(f"  nlist={nlist}, M={M}, nbits={nbits}")

        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFPQ(
            quantizer, dimension, nlist, M, nbits
        )

        # IVFPQ needs more training data than IVFFlat
        # Rule: at least 39 * nlist vectors for training
        min_train = 39 * nlist
        train_data = embeddings
        if len(embeddings) < min_train:
            # If not enough vectors, repeat existing ones for training
            repeats = (min_train // len(embeddings)) + 1
            train_data = np.tile(embeddings, (repeats, 1))[:min_train]

        print("  Training index (this may take a minute)...")
        index.train(train_data)
        index.add(embeddings)
        index.nprobe = 32
        print(f"  nprobe = {index.nprobe}")

    print(f"\nFAISS index built: {index.ntotal} vectors")

    # Save
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index saved → {INDEX_PATH}")
    print(f"Chunks saved → {CHUNKS_PATH}")

# ─────────────────────────────────────────
# LOAD INDEX
# ─────────────────────────────────────────

def load_index():
    """
    Load FAISS index and chunks from disk.
    Returns (index, chunks) tuple.
    Called at app startup — not on every query.
    """
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "No index found. Run build_index() first by running vectorstore.py"
        )

    index = faiss.read_index(str(INDEX_PATH))

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"Index loaded: {index.ntotal} vectors, {len(chunks)} chunks")
    return index, chunks


# ─────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────

def search(
    query: str,
    index,
    chunks: List[Dict],
    top_k: int = 20
) -> List[Dict]:
    """
    Search the FAISS index.
    nprobe is saved inside the index object, 
    so no extra config needed here.
    """
    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    from ingest import ingest_documents

    # Step 1: Ingest documents
    chunks = ingest_documents()
    if not chunks:
        print("No chunks found. Add documents to backend/data/")
        exit()

    # Step 2: Build and save index
    build_index(chunks)

    # Step 3: Test a search
    print("\n--- Test Search ---")
    index, chunks = load_index()

    query = "what is water cement ratio"
    results = search(query, index, chunks, top_k=5)

    print(f"\nQuery: '{query}'")
    print(f"Top {len(results)} results:\n")
    for i, r in enumerate(results):
        print(f"[{i+1}] Score: {r['score']:.4f}")
        print(f"     Text: {r['text'][:100]}...")
        print(f"     Source: {r['source']}")
        print()