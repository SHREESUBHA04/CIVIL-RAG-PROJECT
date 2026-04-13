import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DATASETS_DIR = Path(__file__).parent.parent / "datasets"

# ─────────────────────────────────────────
# LOAD ALL CSV FILES AT STARTUP
# ─────────────────────────────────────────

def load_datasets(datasets_dir: Path = DATASETS_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from the datasets folder into memory.
    Returns a dict: filename → DataFrame

    Called once at startup. DataFrames stay in memory.
    Much faster than re-reading CSVs on every query.
    """
    datasets = {}

    if not datasets_dir.exists():
        print(f"  ⚠ Datasets folder not found: {datasets_dir}")
        return datasets

    csv_files = list(datasets_dir.glob("*.csv"))
    if not csv_files:
        print(f"  ⚠ No CSV files found in {datasets_dir}")
        return datasets

    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
            datasets[filepath.name] = df
            print(f"  ✓ Loaded {filepath.name}: {df.shape[0]} rows × {df.shape[1]} cols")
        except Exception as e:
            print(f"  ✗ Failed to load {filepath.name}: {e}")

    return datasets


# ─────────────────────────────────────────
# COMPUTE STATISTICS
# ─────────────────────────────────────────

def compute_statistics(datasets: Dict[str, pd.DataFrame]) -> Dict:
    """
    Pre-compute statistics for all datasets.
    Returns a structured summary the LLM can reason about.

    We compute this once and cache it — not on every query.
    """
    stats = {}

    for name, df in datasets.items():
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        dataset_stats = {
            "filename": name,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "numeric_columns": numeric_cols,
            "summary": {}
        }

        for col in numeric_cols:
            series = df[col].dropna()
            dataset_stats["summary"][col] = {
                "mean":   round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std":    round(float(series.std()), 4),
                "min":    round(float(series.min()), 4),
                "max":    round(float(series.max()), 4),
                "count":  int(series.count()),
            }

        # Correlation matrix for numeric columns
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().round(4)
            dataset_stats["correlations"] = corr.to_dict()

        stats[name] = dataset_stats

    return stats


def format_stats_for_llm(stats: Dict) -> str:
    """
    Format the statistics dict into a clean string the LLM can read.
    Keeps it concise — LLMs don't need raw DataFrames, just the numbers.
    """
    parts = []

    for name, ds in stats.items():
        lines = [f"Dataset: {name} ({ds['rows']} rows)"]
        lines.append(f"Columns: {', '.join(ds['columns'])}")
        lines.append("\nColumn Statistics:")

        for col, s in ds["summary"].items():
            lines.append(
                f"  {col}: mean={s['mean']}, median={s['median']}, "
                f"std={s['std']}, min={s['min']}, max={s['max']}, "
                f"count={s['count']}"
            )

        if "correlations" in ds:
            lines.append("\nTop Correlations (|r| > 0.5):")
            corr_dict = ds["correlations"]
            cols = list(corr_dict.keys())
            seen = set()
            for c1 in cols:
                for c2 in cols:
                    if c1 != c2 and (c2, c1) not in seen:
                        r = corr_dict[c1].get(c2, 0)
                        if abs(r) > 0.5:
                            lines.append(f"  {c1} ↔ {c2}: r={r:.4f}")
                        seen.add((c1, c2))

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


# ─────────────────────────────────────────
# ANSWER DATA QUESTIONS
# ─────────────────────────────────────────

DATA_PROMPT = """You are a civil engineering data analyst.
Answer the user's question using ONLY the dataset statistics provided below.

STRICT RULES:
- Base your answer only on the provided statistics
- Quote specific numbers from the data
- If the question cannot be answered from the statistics, say so clearly
- Keep the answer technical and precise
- Mention the dataset name in your answer

DATASET STATISTICS:
{stats}

USER QUESTION:
{query}

ANSWER:"""


def answer_data_question(
    query: str,
    stats: Dict,
    conversation_history: List[Dict] = None
) -> Dict:
    """
    Answer a data-related question using pre-computed statistics.

    Args:
        query                : user's data question
        stats                : pre-computed statistics dict
        conversation_history : previous conversation for context

    Returns dict with answer and metadata.
    """
    if not stats:
        return {
            "answer": "No datasets are loaded. Please add CSV files to the datasets/ folder.",
            "sources": [],
            "chunks": []
        }

    stats_text = format_stats_for_llm(stats)

    messages = []
    if conversation_history:
        messages.extend(conversation_history[-12:])

    messages.append({
        "role": "user",
        "content": DATA_PROMPT.format(stats=stats_text, query=query)
    })

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=600,
        )

        answer = response.choices[0].message.content.strip()
        sources = list(stats.keys())

        return {
            "answer": answer,
            "sources": sources,
            "chunks": [],
            "data_stats": stats_text
        }

    except Exception as e:
        return {
            "answer": f"Data analysis failed: {str(e)}",
            "sources": [],
            "chunks": []
        }


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Loading datasets...")
    datasets = load_datasets()

    if not datasets:
        print("\n⚠ No datasets found.")
        print("Add CSV files to backend/datasets/ folder.")
        print("\nCreating a sample dataset for testing...")

        # Create a sample concrete strength dataset for testing
        DATASETS_DIR.mkdir(exist_ok=True)
        sample_data = {
            "cement": [540, 332, 363, 289, 365, 388, 405, 437, 469, 287],
            "water":  [162, 228, 228, 228, 225, 230, 226, 228, 228, 228],
            "w_c_ratio": [0.30, 0.69, 0.63, 0.79, 0.62, 0.59, 0.56, 0.52, 0.49, 0.79],
            "age_days":  [28, 28, 28, 28, 28, 28, 28, 28, 28, 28],
            "strength_mpa": [79.99, 26.25, 34.99, 18.13, 36.45, 45.08, 43.70, 54.36, 63.53, 18.13],
        }
        df = pd.DataFrame(sample_data)
        sample_path = DATASETS_DIR / "concrete_strength.csv"
        df.to_csv(sample_path, index=False)
        print(f"Sample dataset created: {sample_path}")
        datasets = load_datasets()

    print("\nComputing statistics...")
    stats = compute_statistics(datasets)

    print("\n--- Testing Data Questions ---")
    test_queries = [
        "what is the average concrete strength in the dataset",
        "what is the correlation between water cement ratio and strength",
        "which sample has the maximum strength",
    ]

    for query in test_queries:
        print(f"\n{'─'*50}")
        print(f"Q: {query}")
        result = answer_data_question(query, stats)
        print(f"A: {result['answer']}")