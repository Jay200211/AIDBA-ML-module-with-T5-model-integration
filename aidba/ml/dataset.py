"""Dataset loading and preprocessing for T5 training.

Uses the Spider dataset (10K+ questions with SQL queries)
for fine-tuning the model to generate SQL.
"""
import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random

log = logging.getLogger("aidba.ml.dataset")


class SQLExample:
    """A single training example."""

    def __init__(self, question: str, sql: str, schema: str = "",
                 database: str = "", db_id: str = ""):
        self.question = question
        self.sql = sql
        self.schema = schema
        self.database = database
        self.db_id = db_id

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "sql": self.sql,
            "schema": self.schema,
            "database": self.database,
            "db_id": self.db_id
        }


class SpiderDataset:
    """Spider dataset loader for NL-to-SQL training."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.examples: List[SQLExample] = []

    def load_from_spider_json(self, json_path: str) -> List[SQLExample]:
        """Load from Spider format JSON file.

        Spider format:
        {
            "question": "What are the names...",
            "query": "SELECT name FROM ...",
            "db_id": "concert_singer",
            "sql": {...}  // parsed SQL
        }
        """
        examples = []
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                example = SQLExample(
                    question=item.get("question", ""),
                    sql=item.get("query", ""),
                    database=item.get("db_id", ""),
                    db_id=item.get("db_id", "")
                )
                if example.question and example.sql:
                    examples.append(example)

            log.info(f"Loaded {len(examples)} examples from {json_path}")
            self.examples = examples
            return examples
        except Exception as e:
            log.exception(f"Failed to load Spider JSON: {e}")
            return []

    def load_from_your_db(self, db_path: str, limit: int = 1000) -> List[SQLExample]:
        """Generate training examples from your own database.

        This creates question-SQL pairs by inspecting your schema
        and generating common query patterns.
        """
        examples = []
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Get all tables
            cur.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = [row["name"] for row in cur.fetchall()]

            for table in tables:
                # Get columns
                cur.execute(f"PRAGMA table_info({table})")
                columns = [row["name"] for row in cur.fetchall()]

                if not columns:
                    continue

                # Generate common query patterns
                for col in columns:
                    # COUNT queries
                    examples.append(SQLExample(
                        question=f"How many {table} are there?",
                        sql=f"SELECT COUNT(*) FROM {table};",
                        database=table
                    ))
                    examples.append(SQLExample(
                        question=f"Count all {table}",
                        sql=f"SELECT COUNT(*) FROM {table};",
                        database=table
                    ))

                # SELECT * queries
                examples.append(SQLExample(
                    question=f"Show all {table}",
                    sql=f"SELECT * FROM {table};",
                    database=table
                ))
                examples.append(SQLExample(
                    question=f"List all {table}",
                    sql=f"SELECT * FROM {table};",
                    database=table
                ))

                # Column-specific queries
                for col in columns:
                    examples.append(SQLExample(
                        question=f"Show {col} from {table}",
                        sql=f"SELECT {col} FROM {table};",
                        database=table
                    ))

                # WHERE queries (find distinct values)
                cur.execute(f"SELECT DISTINCT {col} FROM {table} LIMIT 5")
                distinct_vals = [row[0] for row in cur.fetchall()]

                for val in distinct_vals:
                    if val and isinstance(val, (str, int)):
                        val_str = f"'{val}'" if isinstance(val, str) else str(val)
                        examples.append(SQLExample(
                            question=f"Show {table} where {col} is {val}",
                            sql=f"SELECT * FROM {table} WHERE {col} = {val_str};",
                            database=table
                        ))

                # Top N queries
                examples.append(SQLExample(
                    question=f"Top 10 {table}",
                    sql=f"SELECT * FROM {table} LIMIT 10;",
                    database=table
                ))

                # Aggregation queries
                numeric_cols = []
                for col in columns:
                    cur.execute(f"SELECT typeof({col}) FROM {table} LIMIT 1")
                    col_type = cur.fetchone()
                    if col_type and ("INT" in str(col_type).upper() or "REAL" in str(col_type).upper() or "NUM" in str(col_type).upper()):
                        numeric_cols.append(col)

                for col in numeric_cols:
                    examples.append(SQLExample(
                        question=f"Average {col} in {table}",
                        sql=f"SELECT AVG({col}) FROM {table};",
                        database=table
                    ))
                    examples.append(SQLExample(
                        question=f"Sum of {col} from {table}",
                        sql=f"SELECT SUM({col}) FROM {table};",
                        database=table
                    ))
                    examples.append(SQLExample(
                        question=f"Maximum {col} in {table}",
                        sql=f"SELECT MAX({col}) FROM {table};",
                        database=table
                    ))

            conn.close()
            log.info(f"Generated {len(examples)} examples from your database")
            self.examples = examples[:limit]
            return self.examples
        except Exception as e:
            log.exception(f"Failed to load from your DB: {e}")
            return []

    def split(self, train_ratio: float = 0.8) -> Tuple[List[SQLExample], List[SQLExample]]:
        """Split into train and eval sets."""
        random.shuffle(self.examples)
        split_idx = int(len(self.examples) * train_ratio)
        return self.examples[:split_idx], self.examples[split_idx:]

    def save_jsonl(self, examples: List[SQLExample], path: str):
        """Save examples to JSONL format."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex.to_dict()) + "\n")
            log.info(f"Saved {len(examples)} examples to {path}")
        except Exception as e:
            log.exception(f"Failed to save JSONL: {e}")

    def load_jsonl(self, path: str) -> List[SQLExample]:
        """Load examples from JSONL format."""
        examples = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    examples.append(SQLExample(
                        question=data["question"],
                        sql=data["sql"],
                        schema=data.get("schema", ""),
                        database=data.get("database", ""),
                        db_id=data.get("db_id", "")
                    ))
            log.info(f"Loaded {len(examples)} examples from {path}")
            return examples
        except Exception as e:
            log.exception(f"Failed to load JSONL: {e}")
            return []
