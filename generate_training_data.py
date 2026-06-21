"""Generate training data from your SQL Server database."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from aidba.ml.dataset import SpiderDataset

# Connect to your SQL Server
db_path = "DRIVER={ODBC Driver 18 for SQL Server};SERVER=JAYENDRA\\SQLEXPRESS;DATABASE=MyDatabase;Trusted_Connection=yes;TrustServerCertificate=yes;"

# Actually, for SQLite generation, let's use the schema
# OR we can directly load from a real Spider dataset

# Option 1: Generate from your own DB
# We need to use a different approach for SQL Server
# Let's just create the training script structure

print("To train the T5 model, you need to either:")
print("1. Download the Spider dataset (https://drive.google.com/uc?id=1iTzV3iQvg2cVc5WPCTdW3oTl9KrF1gSP)")
print("2. Or generate examples from your own database")
print()
print("For now, let's create a simple synthetic dataset...")

# Create a simple synthetic dataset
import json
from pathlib import Path

synthetic_data = [
    {"question": "How many customers are there?", "sql": "SELECT COUNT(*) FROM Customers;", "database": "test"},
    {"question": "Show all customers", "sql": "SELECT * FROM Customers;", "database": "test"},
    {"question": "List customer names", "sql": "SELECT first_name FROM Customers;", "database": "test"},
    {"question": "Show customers from Germany", "sql": "SELECT * FROM Customers WHERE country = 'Germany';", "database": "test"},
    {"question": "Count orders", "sql": "SELECT COUNT(*) FROM Orders;", "database": "test"},
    {"question": "Show all orders", "sql": "SELECT * FROM Orders;", "database": "test"},
    {"question": "List order IDs", "sql": "SELECT order_id FROM Orders;", "database": "test"},
    {"question": "Show customers with score above 500", "sql": "SELECT * FROM Customers WHERE score > 500;", "database": "test"},
    {"question": "Top 10 customers by score", "sql": "SELECT * FROM Customers ORDER BY score DESC LIMIT 10;", "database": "test"},
    {"question": "Average customer score", "sql": "SELECT AVG(score) FROM Customers;", "database": "test"},
    {"question": "Delete all customers from USA", "sql": "DELETE FROM Customers WHERE country = 'USA';", "database": "test"},
    {"question": "Update customer score to 100 for Germany", "sql": "UPDATE Customers SET score = 100 WHERE country = 'Germany';", "database": "test"},
    {"question": "Insert a new customer", "sql": "INSERT INTO Customers (first_name, country, score) VALUES ('New', 'USA', 500);", "database": "test"},
    {"question": "Drop the test table", "sql": "DROP TABLE test_table;", "database": "test"},
]

# Save as JSONL
data_dir = Path("D:/aidba/data/training")
data_dir.mkdir(parents=True, exist_ok=True)

train_path = data_dir / "train.jsonl"
with open(train_path, "w", encoding="utf-8") as f:
    for item in synthetic_data[:10]:  # First 10 for training
        f.write(json.dumps(item) + "\n")

eval_path = data_dir / "eval.jsonl"
with open(eval_path, "w", encoding="utf-8") as f:
    for item in synthetic_data[10:]:  # Last 4 for eval
        f.write(json.dumps(item) + "\n")

print(f"✅ Created {len(synthetic_data[:10])} training examples")
print(f"✅ Created {len(synthetic_data[10:])} eval examples")
print(f"📁 Training data: {train_path}")
print(f"📁 Eval data: {eval_path}")
