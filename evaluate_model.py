"""Evaluate the trained model's accuracy on your test data."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from aidba.ml.predict import SQLPredictor
from aidba.ml.dataset import SpiderDataset

print("=" * 60)
print("Model Accuracy Evaluation")
print("=" * 60)

# Load model
predictor = SQLPredictor(model_path="D:\\aidba\\models\\aidba-sql-t5")

if not predictor.loaded:
    print("❌ Model failed to load")
    sys.exit(1)

# Load test data
dataset = SpiderDataset()
test_data = dataset.load_jsonl("D:\\aidba\\data\\training\\eval.jsonl")

if not test_data:
    print("❌ No test data found")
    sys.exit(1)

print(f"Test examples: {len(test_data)}\n")

# Test each example
correct = 0
syntax_valid = 0
total = len(test_data)

for i, ex in enumerate(test_data, 1):
    predicted = predictor.predict(ex.question)
    gold = ex.sql

    # Check syntax (basic)
    if predicted and any(kw in predicted.upper() for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
        syntax_valid += 1

    # Check exact match
    is_match = predicted and predicted.strip().lower() == gold.strip().lower()
    if is_match:
        correct += 1

    status = "✅" if is_match else "❌"
    print(f"[{i}] {status} Q: {ex.question[:50]}...")
    if not is_match:
        print(f"    Expected: {gold[:80]}")
        print(f"    Got:      {predicted[:80] if predicted else 'None'}")
    print()

print("=" * 60)
print(f"Total:        {total}")
print(f"Syntax Valid: {syntax_valid}/{total} ({100*syntax_valid/total:.1f}%)")
print(f"Exact Match:  {correct}/{total} ({100*correct/total:.1f}%)")
print("=" * 60)
