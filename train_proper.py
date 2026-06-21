"""Generate larger training dataset for better T5 model accuracy.

This creates ~1000+ synthetic training examples covering common SQL patterns.
"""
import sys
import json
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from aidba.ml.dataset import SpiderDataset, SQLExample

random.seed(42)

# ============================================
# Generate Training Data
# ============================================
print("📊 Generating training dataset...")
examples = []

# Table templates
TABLE_TEMPLATES = {
    "customers": ["id", "first_name", "last_name", "email", "country", "city", "age", "score", "status", "created_at"],
    "orders": ["order_id", "customer_id", "product_id", "amount", "sales", "quantity", "order_date", "status", "region"],
    "products": ["product_id", "name", "category", "price", "stock", "description"],
    "employees": ["employee_id", "name", "department", "salary", "hire_date", "manager_id"],
    "users": ["user_id", "username", "email", "created_at", "last_login", "status"],
}

COUNTRIES = ["USA", "UK", "Germany", "France", "Japan", "Canada", "Australia"]
CITIES = ["New York", "London", "Berlin", "Paris", "Tokyo", "Toronto", "Sydney"]
STATUSES = ["active", "inactive", "pending", "completed", "cancelled"]
REGIONS = ["North", "South", "East", "West", "Central"]
DEPARTMENTS = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]
CATEGORIES = ["Electronics", "Clothing", "Food", "Books", "Toys"]


def add_examples(examples_list, question, sql, table):
    """Helper to add examples."""
    examples_list.append(SQLExample(
        question=question,
        sql=sql,
        database=table
    ))


# ============================================
# 1. COUNT queries (200 examples)
# ============================================
for table in TABLE_TEMPLATES:
    for _ in range(35):
        templates = [
            f"How many {table} are there?",
            f"Count all {table}",
            f"Total number of {table}",
            f"Number of {table} records",
            f"How many records in {table}?",
            f"Count {table}",
            f"Total {table}",
            f"How many {table} are in the database?",
        ]
        q = random.choice(templates)
        add_examples(examples, q, f"SELECT COUNT(*) FROM {table};", table)

# ============================================
# 2. SELECT * with WHERE (300 examples)
# ============================================
for table, columns in TABLE_TEMPLATES.items():
    # Country filter
    if "country" in columns:
        for val in COUNTRIES:
            for _ in range(2):
                templates = [
                    f"Show {table} from {val}",
                    f"List all {table} where country is {val}",
                    f"Get {table} with country {val}",
                    f"Show me {table} from {val}",
                    f"Display {table} from {val}",
                ]
                q = random.choice(templates)
                add_examples(examples, q,
                    f"SELECT * FROM {table} WHERE country = '{val}';", table)

    # Status filter
    if "status" in columns:
        for val in STATUSES:
            for _ in range(2):
                templates = [
                    f"Show {table} with status {val}",
                    f"List {table} where status is {val}",
                    f"Get {table} having status {val}",
                    f"Show me {table} with status {val}",
                ]
                q = random.choice(templates)
                add_examples(examples, q,
                    f"SELECT * FROM {table} WHERE status = '{val}';", table)

    # Region filter
    if "region" in columns:
        for val in REGIONS:
            for _ in range(2):
                templates = [
                    f"Show {table} from {val} region",
                    f"List {table} in {val} region",
                    f"Get {table} where region is {val}",
                ]
                q = random.choice(templates)
                add_examples(examples, q,
                    f"SELECT * FROM {table} WHERE region = '{val}';", table)

    # Department filter
    if "department" in columns:
        for val in DEPARTMENTS:
            for _ in range(2):
                templates = [
                    f"Show {table} in {val} department",
                    f"List {table} where department is {val}",
                    f"Get {table} from {val} department",
                ]
                q = random.choice(templates)
                add_examples(examples, q,
                    f"SELECT * FROM {table} WHERE department = '{val}';", table)

# ============================================
# 3. COUNT with WHERE (200 examples)
# ============================================
for table, columns in TABLE_TEMPLATES.items():
    if "country" in columns:
        for val in COUNTRIES[:5]:
            for _ in range(2):
                templates = [
                    f"Count {table} from {val}",
                    f"How many {table} are from {val}?",
                    f"Number of {table} in {val}",
                ]
                q = random.choice(templates)
                add_examples(examples, q,
                    f"SELECT COUNT(*) FROM {table} WHERE country = '{val}';", table)

    if "status" in columns:
        for val in STATUSES[:3]:
            for _ in range(2):
                templates = [
                    f"Count {table} with status {val}",
                    f"How many {table} have status {val}?",
                ]
                q = random.choice(templates)
                add_examples(examples, q,
                    f"SELECT COUNT(*) FROM {table} WHERE status = '{val}';", table)

# ============================================
# 4. Numeric WHERE (150 examples)
# ============================================
for table, columns in TABLE_TEMPLATES.items():
    for col in ["score", "age", "salary", "price", "amount", "sales", "quantity"]:
        if col in columns:
            for threshold in [100, 500, 1000, 50, 1000]:
                for _ in range(2):
                    templates = [
                        f"Show {table} with {col} above {threshold}",
                        f"Find {table} where {col} > {threshold}",
                        f"Get {table} having {col} greater than {threshold}",
                        f"Count {table} with {col} > {threshold}",
                        f"List {table} where {col} over {threshold}",
                    ]
                    q = random.choice(templates)
                    is_count = "count" in q or "Count" in q or "many" in q
                    sql_op = f"SELECT COUNT(*) FROM {table} WHERE {col} > {threshold};" if is_count \
                        else f"SELECT * FROM {table} WHERE {col} > {threshold};"
                    add_examples(examples, q, sql_op, table)

# ============================================
# 5. SELECT all (100 examples)
# ============================================
for table in TABLE_TEMPLATES:
    for _ in range(18):
        templates = [
            f"Show all {table}",
            f"List all {table}",
            f"Get all {table}",
            f"Display all {table}",
            f"Show me the {table}",
            f"Show every {table}",
        ]
        q = random.choice(templates)
        add_examples(examples, q, f"SELECT * FROM {table};", table)

# ============================================
# 6. Aggregation (100 examples)
# ============================================
for table, columns in TABLE_TEMPLATES.items():
    for col in ["score", "amount", "sales", "price", "salary", "quantity"]:
        if col in columns:
            for _ in range(3):
                templates = [
                    f"What is the average {col} in {table}?",
                    f"Sum of {col} from {table}",
                    f"Total {col} in {table}",
                    f"Maximum {col} in {table}",
                    f"Minimum {col} from {table}",
                    f"Avg {col} of {table}",
                ]
                q = random.choice(templates)
                if "average" in q or "avg" in q.lower():
                    sql = f"SELECT AVG({col}) FROM {table};"
                elif "sum" in q.lower() or "total" in q.lower():
                    sql = f"SELECT SUM({col}) FROM {table};"
                elif "maximum" in q.lower() or "max" in q.lower():
                    sql = f"SELECT MAX({col}) FROM {table};"
                elif "minimum" in q.lower() or "min" in q.lower():
                    sql = f"SELECT MIN({col}) FROM {table};"
                else:
                    sql = f"SELECT AVG({col}) FROM {table};"
                add_examples(examples, q, sql, table)

# ============================================
# 7. Specific columns (150 examples)
# ============================================
column_queries = [
    ("Show names from {table}", "SELECT name FROM {table};"),
    ("List emails from {table}", "SELECT email FROM {table};"),
    ("Get {col} from {table}", "SELECT {col} FROM {table};"),
    ("Show {col} for all {table}", "SELECT {col} FROM {table};"),
    ("Display {col} of {table}", "SELECT {col} FROM {table};"),
]
for table, columns in TABLE_TEMPLATES.items():
    for col in columns[:3]:
        for _ in range(2):
            q_template, sql_template = random.choice(column_queries)
            if "{col}" in q_template:
                q = q_template.format(table=table, col=col)
                sql = sql_template.format(table=table, col=col)
            else:
                q = q_template.format(table=table, col=col)
                sql = sql_template.format(table=table)
            add_examples(examples, q, sql, table)

# ============================================
# 8. Top N (50 examples)
# ============================================
for table in TABLE_TEMPLATES:
    for n in [5, 10, 20, 50]:
        for _ in range(3):
            templates = [
                f"Show top {n} {table}",
                f"Get first {n} {table}",
                f"Top {n} {table} by id",
                f"Show {n} {table}",
            ]
            q = random.choice(templates)
            add_examples(examples, q,
                f"SELECT TOP {n} * FROM {table};", table)

# Shuffle
random.shuffle(examples)
print(f"✅ Generated {len(examples)} training examples")

# Split
split = int(len(examples) * 0.8)
train_examples = examples[:split]
eval_examples = examples[split:]

# Save
data_dir = Path("D:/aidba/data/training")
data_dir.mkdir(parents=True, exist_ok=True)

train_path = data_dir / "train.jsonl"
with open(train_path, "w", encoding="utf-8") as f:
    for ex in train_examples:
        f.write(json.dumps(ex.to_dict()) + "\n")
print(f"✅ Saved {len(train_examples)} training examples to {train_path}")

eval_path = data_dir / "eval.jsonl"
with open(eval_path, "w", encoding="utf-8") as f:
    for ex in eval_examples:
        f.write(json.dumps(ex.to_dict()) + "\n")
print(f"✅ Saved {len(eval_examples)} eval examples to {eval_path}")

print()
print("=" * 60)
print("✅ Training data ready!")
print("=" * 60)
print()
print("Next step - train the model (will take 30-60 minutes):")
print()
print(f'  python -m aidba.ml.train --train-data "{train_path}" --eval-data "{eval_path}" --output-dir "D:\\aidba\\models\\aidba-sql-t5" --base-model t5-small --epochs 5 --batch-size 8')
print()
print("=" * 60)
