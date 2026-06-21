"""Evaluation script for the trained T5 model.

Computes accuracy, exact match, and execution match metrics.
"""
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aidba.ml.evaluate")


def evaluate_model(model_path: str, eval_data_path: str, db_path: str = None):
    """Evaluate the trained T5 model.

    Metrics:
    - Exact Match Accuracy: Does predicted SQL exactly match gold SQL?
    - Execution Accuracy: Does predicted SQL return same results as gold?
    - Syntax Validity: Is predicted SQL syntactically valid?
    """
    try:
        from aidba.ml.model import T5SQLModel
        from aidba.ml.dataset import SpiderDataset
    except ImportError as e:
        log.error(f"Import error: {e}")
        return

    # Load model
    log.info(f"Loading model from: {model_path}")
    model = T5SQLModel(model_path=model_path)
    if not model.load():
        log.error("Failed to load model")
        return

    # Load eval data
    log.info(f"Loading eval data from: {eval_data_path}")
    dataset = SpiderDataset()
    if eval_data_path.endswith(".jsonl"):
        examples = dataset.load_jsonl(eval_data_path)
    else:
        examples = dataset.load_from_spider_json(eval_data_path)

    if not examples:
        log.error("No examples loaded!")
        return

    log.info(f"Evaluating on {len(examples)} examples")

    # Connect to test database if provided
    conn = None
    if db_path:
        try:
            conn = sqlite3.connect(db_path)
            log.info(f"Connected to test database: {db_path}")
        except Exception as e:
            log.warning(f"Could not connect to test DB: {e}")

    # Evaluate
    exact_match = 0
    execution_match = 0
    syntax_valid = 0
    total = len(examples)

    results = []

    for i, ex in enumerate(examples):
        if i % 50 == 0:
            log.info(f"Progress: {i}/{total}")

        # Generate prediction
        predicted_sql = model.generate_sql(ex.question, ex.schema)

        if not predicted_sql:
            results.append({"question": ex.question, "predicted": "", "gold": ex.sql, "match": False})
            continue

        # Check syntax validity (simple check)
        if "SELECT" in predicted_sql.upper() or "INSERT" in predicted_sql.upper() \
           or "UPDATE" in predicted_sql.upper() or "DELETE" in predicted_sql.upper():
            syntax_valid += 1

        # Exact match
        if normalize_sql(predicted_sql) == normalize_sql(ex.sql):
            exact_match += 1

        # Execution match
        if conn and ex.database:
            try:
                pred_result = conn.execute(predicted_sql).fetchall()
                gold_result = conn.execute(ex.sql).fetchall()
                if pred_result == gold_result:
                    execution_match += 1
            except Exception:
                pass

        results.append({
            "question": ex.question,
            "predicted": predicted_sql,
            "gold": ex.sql,
            "exact_match": normalize_sql(predicted_sql) == normalize_sql(ex.sql)
        })

    if conn:
        conn.close()

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total examples:      {total}")
    print(f"Syntax valid:        {syntax_valid} ({100*syntax_valid/total:.1f}%)")
    print(f"Exact match:         {exact_match} ({100*exact_match/total:.1f}%)")
    if conn:
        print(f"Execution match:     {execution_match} ({100*execution_match/total:.1f}%)")
    print("=" * 60)

    # Save detailed results
    output_path = Path(model_path).parent / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "syntax_valid_pct": 100 * syntax_valid / total,
            "exact_match_pct": 100 * exact_match / total,
            "execution_match_pct": 100 * execution_match / total if conn else None,
            "results": results[:100]  # First 100 examples
        }, f, indent=2)
    log.info(f"Detailed results saved to: {output_path}")


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    import re
    sql = sql.lower().strip()
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.rstrip(';')
    return sql


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate T5 SQL model")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--eval-data", required=True, help="Path to eval data")
    parser.add_argument("--db-path", help="Path to test database (for execution match)")

    args = parser.parse_args()
    evaluate_model(args.model_path, args.eval_data, args.db_path)
