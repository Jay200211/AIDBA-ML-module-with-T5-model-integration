"""Test the trained T5 model directly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from aidba.ml.predict import SQLPredictor

print("=" * 60)
print("Testing T5 SQL Model")
print("=" * 60)

# Load the predictor
predictor = SQLPredictor(model_path="D:\\aidba\\models\\aidba-sql-t5")

if not predictor.loaded:
    print("❌ Model failed to load!")
    sys.exit(1)

print(f"✅ Model loaded successfully")
print()

# Test questions
test_questions = [
    "How many customers are there?",
    "Show all customers",
    "List customer names",
    "Show customers from Germany",
    "Count all orders",
    "Show orders with sales above 20",
    "Top 10 customers by score",
]

for i, question in enumerate(test_questions, 1):
    print(f"[{i}] Q: {question}")
    sql = predictor.predict(question)
    print(f"    SQL: {sql}")
    print()

print("=" * 60)
print("Test complete!")
