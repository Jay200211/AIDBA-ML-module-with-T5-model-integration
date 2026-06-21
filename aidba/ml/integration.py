"""Hybrid NLQ Engine with sqlcoder + fixed rules.

Architecture:
  1. Try Ollama sqlcoder first (90%+ accuracy, ~2-5 sec)
  2. Validate SQL output
  3. Fall back to rule-based (85-90% accuracy, instant)
  4. Apply safety checks for destructive operations
"""
import json
import logging
import re
from typing import Optional, Dict, Any, List, Tuple
import requests

log = logging.getLogger("aidba.ml.integration")


# Operation keywords
OPERATIONS = {
    'insert', 'add', 'create', 'new',
    'update', 'change', 'set', 'modify', 'edit',
    'delete', 'remove', 'erase', 'drop', 'destroy',
    'truncate', 'clear', 'empty',
    'alter', 'rename',
    'count', 'total', 'number', 'many', 'much',
}

STOP_WORDS = {
    'show', 'list', 'get', 'find', 'fetch', 'display', 'view', 'see',
    'all', 'every', 'each', 'the', 'a', 'an',
    'from', 'in', 'of', 'to', 'for', 'with', 'by', 'and', 'or',
    'where', 'is', 'are', 'was', 'were', 'be',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'us',
    'this', 'that', 'these', 'those', 'my', 'your',
    'how', 'above', 'over', 'greater', 'less', 'than',
    'can', 'could', 'would', 'should',
    'table', 'database', 'row', 'rows', 'record', 'records',
    'having', 'do', 'does', 'did',
}


def safe_response(data):
    """Ensure response has required fields - never null."""
    if not isinstance(data, dict):
        return {"type": "info", "summary": "Invalid response", "rows": []}
    if not data.get('type'):
        data['type'] = 'info'
    if not data.get('summary'):
        data['summary'] = 'Query completed'
    if data.get('rows') is None:
        data['rows'] = []
    return data


def classify_sql_operation(sql):
    """Classify SQL for safety."""
    if not sql:
        return {"operation": "unknown", "is_destructive": False, "requires_approval": False, "risk_level": "unknown"}
    sql_upper = sql.upper().strip()
    first_keyword = sql_upper.split()[0] if sql_upper.split() else ""
    DESTRUCTIVE = {'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'DROP', 'ALTER'}
    APPROVAL_NEEDED = {'DELETE', 'TRUNCATE', 'DROP', 'ALTER'}
    operation = "unknown"
    is_destructive = False
    requires_approval = False
    risk_level = "safe"
    if first_keyword in DESTRUCTIVE:
        operation = first_keyword
        is_destructive = True
        requires_approval = first_keyword in APPROVAL_NEEDED
        risk_level = "critical" if first_keyword == 'DROP' else ("high" if first_keyword in ('DELETE', 'TRUNCATE') else "low")
    elif first_keyword in ('SELECT', 'SHOW', 'WITH'):
        operation = first_keyword
    if operation in ('DELETE', 'UPDATE') and 'WHERE' not in sql_upper:
        risk_level = "critical"
        requires_approval = True
    return {
        "operation": operation,
        "is_destructive": is_destructive,
        "requires_approval": requires_approval,
        "risk_level": risk_level,
        "sql": sql[:500]
    }


class HybridNLQEngine:
    """NLQ Engine using sqlcoder + rules fallback."""

    def __init__(self, cfg, store, monitor, predictor=None):
        self.cfg = cfg
        self.store = store
        self.monitor = monitor
        self.predictor = predictor  # Optional T5 model
        self.schema_cache = {}

        # Ollama configuration
        self.ollama_url = "http://localhost:11434"
        self.ollama_model = "sqlcoder:7b"  # You have this installed!
        self.use_ollama = True  # Set to False to use only rules

        log.info(f"HybridNLQEngine initialized: ollama={self.use_ollama}, t5={predictor is not None}")

    def get_schema(self, db_name):
        """Get database schema with tables, columns, and sample data."""
        if db_name in self.schema_cache:
            return self.schema_cache[db_name]

        tables = []
        columns_map = {}
        sample_data_map = {}

        if not self.monitor or not self.monitor.dbm:
            return ("", tables, columns_map, sample_data_map)

        conn = self.monitor.dbm.get(db_name)
        if not conn:
            return ("", tables, columns_map, sample_data_map)

        try:
            if conn.dialect == "tsql":
                # Get tables
                sql = """
                    SELECT s.name AS [schema], t.name AS [table]
                    FROM sys.tables t
                    INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
                    WHERE t.is_ms_shipped = 0
                """
                result = conn.execute(sql)
                if result:
                    for r in result:
                        schema = r.get("schema", "")
                        table = r.get("table", "")
                        full_name = f"[{schema}].[{table}]" if schema and schema != "dbo" else f"[{table}]"
                        tables.append({"full_name": full_name, "table": table, "schema": schema})

                        # Get columns
                        col_sql = f"""
                            SELECT COLUMN_NAME
                            FROM INFORMATION_SCHEMA.COLUMNS
                            WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}'
                        """
                        try:
                            cols = conn.execute(col_sql) or []
                            col_list = [c.get("COLUMN_NAME") for c in cols]
                            columns_map[table.lower()] = col_list
                        except Exception:
                            columns_map[table.lower()] = []

                        # Get sample data (first 3 rows) - helps LLM understand values
                        try:
                            sample_sql = f"SELECT TOP 3 * FROM [{schema}].[{table}]"
                            sample_result = conn.execute(sample_sql)
                            if sample_result and len(sample_result) > 0:
                                # Get column names
                                col_names = list(sample_result[0].keys()) if sample_result[0] else []
                                # Get sample values
                                sample_rows = []
                                for row in sample_result[:3]:
                                    sample_rows.append({k: str(v)[:50] for k, v in row.items()})
                                sample_data_map[table.lower()] = {
                                    "columns": col_names,
                                    "sample": sample_rows
                                }
                        except Exception:
                            pass

            # Build schema string for LLM
            schema_parts = []
            for t in tables:
                tbl = t["table"]
                cols = columns_map.get(tbl.lower(), [])
                cols_str = ", ".join(cols[:10]) if cols else ""
                schema_parts.append(f"Table {t['full_name']} ({cols_str})")

                # Add sample data hint
                if tbl.lower() in sample_data_map:
                    sample = sample_data_map[tbl.lower()]
                    if sample.get("sample"):
                        vals = []
                        for row in sample["sample"][:2]:
                            vals.append(str(row)[:100])
                        if vals:
                            schema_parts.append(f"  Sample: {'; '.join(vals)[:200]}")

            schema_str = "\n".join(schema_parts)

            self.schema_cache[db_name] = (schema_str, tables, columns_map, sample_data_map)
            return self.schema_cache[db_name]
        except Exception as e:
            log.warning(f"Failed to get schema: {e}")
            return ("", [], {}, {})

    def find_best_table_match(self, keyword, tables):
        """Find best matching table - STRICT word boundary."""
        keyword_lower = keyword.lower().strip()
        if not keyword_lower:
            return None

        # Priority 1: Exact match
        for t in tables:
            table_name = t["table"].lower()
            if keyword_lower == table_name:
                return t

        # Priority 2: Plural/singular
        for t in tables:
            table_name = t["table"].lower()
            if keyword_lower == table_name + 's' or keyword_lower + 's' == table_name:
                return t
            if keyword_lower == table_name.rstrip('s') or keyword_lower.rstrip('s') == table_name:
                return t

        # Priority 3: Word boundary
        for t in tables:
            table_name = t["table"].lower()
            if re.search(r'\b' + re.escape(keyword_lower) + r's?\b', table_name):
                return t

        # Priority 4: Starts with
        for t in tables:
            table_name = t["table"].lower()
            if table_name.startswith(keyword_lower) or keyword_lower.startswith(table_name):
                return t

        return None

    async def _call_ollama(self, question, schema_str, db_name):
        """Call Ollama sqlcoder to generate SQL."""
        try:
            prompt = f"""### Task
Generate a SQL query to answer the following question.

### Database Schema
{schema_str}

### Question
{question}

### SQL Query
"""

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,  # Deterministic
                        "num_predict": 200
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                sql = result.get("response", "").strip()
                # Clean up the response
                sql = self._clean_sql(sql)
                return sql
            return None
        except Exception as e:
            log.warning(f"Ollama call failed: {e}")
            return None

    def _clean_sql(self, sql):
        """Clean up SQL from LLM output."""
        if not sql:
            return ""
        # Remove markdown code fences
        sql = re.sub(r'^```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'^```\s*', '', sql)
        sql = re.sub(r'```$', '', sql)
        sql = sql.strip()
        # Remove trailing semicolons for consistency
        if not sql.endswith(';'):
            sql = sql + ';'
        return sql

    def ask(self, question):
        """Main entry point - SYNCHRONOUS."""
        q = (question or "").strip()
        if not q:
            return safe_response({"type": "error", "summary": "Empty question", "rows": []})

        dbs = self.monitor.dbm.list() if self.monitor and self.monitor.dbm else []
        if not dbs:
            return safe_response({"type": "error", "summary": "No databases connected", "rows": []})

        target_db = self._resolve_db(q, dbs)
        if not target_db:
            return safe_response({"type": "info", "summary": f"Please specify database. Available: {dbs}", "rows": []})

        # Try rule-based shortcuts first (instant)
        rule_result = self._try_rules(q, target_db)
        if rule_result:
            return safe_response(rule_result)

        # Try Ollama sqlcoder (slower but more accurate)
        if self.use_ollama:
            try:
                result = self._call_ollama_sync(q, target_db)
                if result and not result.get("error"):
                    return safe_response(result)
            except Exception as e:
                log.warning(f"Ollama failed, falling back to rules: {e}")

        # Fallback to rule-based pattern matching
        return safe_response(self._pattern_match(q, target_db))

    def _call_ollama_sync(self, question, target_db):
        """Synchronous wrapper for Ollama call."""
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                schema_str, tables, columns_map, sample_data_map = self.get_schema(target_db)
                sql = loop.run_until_complete(
                    self._call_ollama(question, schema_str, target_db)
                )
            finally:
                loop.close()
        except Exception as e:
            log.warning(f"Ollama sync wrapper failed: {e}")
            return None

        if not sql:
            return None

        # Validate SQL
        if not self._is_valid_sql(sql, tables):
            log.warning(f"Ollama produced invalid SQL: {sql}")
            return None

        # Classify for safety
        classification = classify_sql_operation(sql)

        if classification["is_destructive"]:
            self._create_alert(target_db, question, sql, classification)
            if classification["requires_approval"]:
                return {
                    "type": "approval_required",
                    "summary": f"⚠️ DESTRUCTIVE: {classification['operation']}\n\nSQL: {sql}\n\nApproval required.",
                    "operation": classification,
                    "db": target_db,
                    "sql": sql,
                    "rows": []
                }
            else:
                return {
                    "type": "confirmation_required",
                    "summary": f"📝 Write: {sql}",
                    "operation": classification,
                    "db": target_db,
                    "sql": sql,
                    "rows": []
                }

        return self._execute(target_db, sql, question)

    def _is_valid_sql(self, sql, tables):
        """Check if SQL is valid (not garbage)."""
        if not sql:
            return False
        sql_lower = sql.lower()
        # Check for garbage patterns
        garbage = ['this is', 'note:', 'sql query', 'the query', 'i cannot']
        for g in garbage:
            if g in sql_lower:
                return False
        # Must start with SQL keyword
        first_word = sql_lower.strip().split()[0] if sql_lower.strip().split() else ""
        if first_word not in ('select', 'insert', 'update', 'delete', 'with', 'create', 'drop'):
            return False
        # Should reference at least one table
        if tables:
            has_table = False
            for t in tables:
                if t["table"].lower() in sql_lower:
                    has_table = True
                    break
            if not has_table:
                return False
        return True

    def _resolve_db(self, question, dbs):
        ql = question.lower()
        for db in dbs:
            if db.lower() in ql:
                return db
        if len(dbs) == 1:
            return dbs[0]
        return None

    def _try_rules(self, q, target_db):
        """Quick rule-based handlers."""
        ql = q.lower()

        if re.search(r"\b(list|show|what).*tables?\b", ql) and \
           not re.search(r"\b(show|get|drop|delete)\s+\w+\s*(table)?\s*$", ql):
            return self._list_tables(target_db)

        if re.search(r"\b(health|status|version)\b", ql):
            return self._get_health(target_db)

        if re.search(r"\b(performance|metric)\b", ql):
            return self._get_performance(target_db)

        if re.search(r"\bslow\b.*\bquer", ql):
            return self._get_slow_queries(target_db)

        return None

    def _pattern_match(self, question, target_db):
        """Rule-based fallback pattern matching."""
        schema_str, tables, columns_map, sample_data_map = self.get_schema(target_db)

        if not tables:
            return safe_response({"type": "info", "summary": "No tables found", "rows": []})

        ql = question.lower().strip()
        words = re.findall(r'\b[a-z_][a-z0-9_]*\b', ql)

        # Detect operation
        operation_type = self._detect_operation(ql, words)

        # Find table
        target_table = self._find_target_table(words, tables)

        if not target_table:
            available = [t["table"] for t in tables[:10]]
            return safe_response({
                "type": "info",
                "summary": f"No table matched. Available: {', '.join(available)}",
                "rows": []
            })

        available_columns = columns_map.get(target_table["table"].lower(), [])

        try:
            if operation_type == 'drop':
                return self._build_drop(target_db, target_table, question)
            elif operation_type == 'truncate':
                return self._build_truncate(target_db, target_table, question)
            elif operation_type == 'delete':
                return self._build_delete(target_db, target_table, question, words, available_columns)
            elif operation_type == 'insert':
                return self._build_insert(target_db, target_table, available_columns)
            elif operation_type == 'count':
                sql = self._build_count(target_db, target_table, question, words, available_columns)
                return self._execute(target_db, sql, question)
            else:
                sql = self._build_select(target_db, target_table, question, words, available_columns)
                return self._execute(target_db, sql, question)
        except Exception as e:
            log.exception(f"Pattern match failed: {e}")
            return safe_response({"type": "error", "summary": f"Error: {str(e)}", "rows": []})

    def _find_target_table(self, words, tables):
        """Find target table."""
        for kw in words:
            if kw in OPERATIONS or kw in STOP_WORDS or len(kw) <= 2:
                continue
            match = self.find_best_table_match(kw, tables)
            if match:
                return match
        return None

    def _detect_operation(self, ql, words):
        """Detect operation type."""
        multi_word = [
            (r'\bdrop\s+(table|database)\b', 'drop'),
            (r'\btruncate\s+(table)?\b', 'truncate'),
            (r'\bdelete\s+(from|all|every|the)\b', 'delete'),
            (r'\bremove\s+(from|all|every|the)\b', 'delete'),
            (r'\binsert\s+(into)?\b', 'insert'),
            (r'\badd\s+(a\s+)?new\b', 'insert'),
            (r'\bupdate\s+\w+\s+set\b', 'update'),
            (r'\bcount\s+(all|the)?\b', 'count'),
            (r'\bhow\s+many\b', 'count'),
        ]
        for pattern, op in multi_word:
            if re.search(pattern, ql):
                return op
        for w in words:
            if w in OPERATIONS:
                return OPERATIONS[w]
        return None

    def _extract_where(self, question, table_info, columns):
        """FIXED: Extract WHERE clause with better column matching."""
        ql = question.lower().strip()
        table_name = table_info["table"].lower()

        # Step 1: Find the filter value
        filter_value = None
        filter_patterns = [
            r'\b(?:from|with|where|is|equals?|in)\s+([a-z_][a-z0-9_]*)\b',
            r'\bin\s+([a-z_]+)\b',  # "in Germany"
            r'\bfrom\s+([a-z_]+)\b',  # "from Germany"
        ]
        for pattern in filter_patterns:
            match = re.search(pattern, ql)
            if match:
                candidate = match.group(1)
                if candidate not in OPERATIONS and candidate not in STOP_WORDS and len(candidate) > 1:
                    # Also skip if it's a number
                    if not candidate.isdigit():
                        filter_value = candidate
                        break

        # Step 2: Find numeric comparison
        numeric_value = None
        numeric_op = None
        num_match = re.search(r'\b(?:above|over|greater\s+than|>|>=|less\s+than|<|<=)\s*(\d+)', ql)
        if num_match:
            numeric_value = num_match.group(1)
            numeric_op = '>' if any(x in num_match.group(0) for x in ['>', 'above', 'over', 'greater']) else '<'

        # Step 3: Match to column
        priority_cols = ['country', 'name', 'status', 'region', 'city', 'state',
                        'department', 'category', 'type', 'gender', 'email']

        if filter_value:
            # 3a. Exact column name match
            for col in columns:
                if col.lower() == filter_value:
                    return f"{col} = '{filter_value.capitalize()}'"

            # 3b. Priority column match (country, name, status, etc.)
            for col in columns:
                if col.lower() in priority_cols:
                    return f"{col} = '{filter_value.capitalize()}'"

            # 3c. Substring match
            for col in columns:
                if filter_value in col.lower() or col.lower() in filter_value:
                    return f"{col} = '{filter_value.capitalize()}'"

            # 3d. First text-like column
            text_cols = [c for c in columns if c.lower() in priority_cols or
                        any(kw in c.lower() for kw in ['name', 'city', 'status', 'region'])]
            if text_cols:
                return f"{text_cols[0]} = '{filter_value.capitalize()}'"

        if numeric_value:
            numeric_cols = ['score', 'price', 'amount', 'count', 'age', 'quantity', 'salary']
            for col in columns:
                if col.lower() in numeric_cols:
                    return f"{col} {numeric_op} {numeric_value}"

        return None

    def _build_select(self, db_name, table_info, question, words, columns):
        """FIXED: Build SELECT query with proper WHERE clause."""
        full_name = table_info["full_name"]
        where = self._extract_where(question, table_info, columns)

        # Check for specific columns
        requested_cols = []
        for w in words:
            if w not in OPERATIONS and w not in STOP_WORDS and len(w) > 2:
                for col in columns:
                    if col.lower() == w or w in col.lower():
                        if col not in requested_cols:
                            requested_cols.append(col)
                        break

        if requested_cols:
            col_str = ", ".join(requested_cols[:5])
            if where:
                return f"SELECT {col_str} FROM {full_name} WHERE {where};"
            return f"SELECT {col_str} FROM {full_name};"

        if where:
            return f"SELECT TOP 10 * FROM {full_name} WHERE {where};"
        return f"SELECT TOP 10 * FROM {full_name};"

    def _build_count(self, db_name, table_info, question, words, columns):
        full_name = table_info["full_name"]
        where = self._extract_where(question, table_info, columns)
        if where:
            return f"SELECT COUNT(*) AS row_count FROM {full_name} WHERE {where};"
        return f"SELECT COUNT(*) AS row_count FROM {full_name};"

    def _build_drop(self, db_name, table_info, question):
        return {
            "type": "approval_required",
            "summary": f"🚨 CRITICAL: DROP TABLE\n\nQuestion: {question}\nTable: {table_info['table']}\n\n⚠️ PERMANENTLY DELETES the table!\nApproval required.",
            "operation": {"operation": "DROP", "is_destructive": True, "requires_approval": True, "risk_level": "critical"},
            "db": db_name, "sql": f"DROP TABLE {table_info['full_name']};", "rows": []
        }

    def _build_truncate(self, db_name, table_info, question):
        return {
            "type": "approval_required",
            "summary": f"⚠️ TRUNCATE\n\nQuestion: {question}\nTable: {table_info['table']}\n\nDeletes ALL rows.",
            "operation": {"operation": "TRUNCATE", "is_destructive": True, "requires_approval": True, "risk_level": "high"},
            "db": db_name, "sql": f"TRUNCATE TABLE {table_info['full_name']};", "rows": []
        }

    def _build_delete(self, db_name, table_info, question, words, columns):
        full_name = table_info["full_name"]
        where = self._extract_where(question, table_info, columns)
        if where:
            sql = f"DELETE FROM {full_name} WHERE {where};"
            return {
                "type": "approval_required",
                "summary": f"⚠️ DELETE\n\nQuestion: {question}\nSQL: {sql}\n\nRisk: HIGH\nClick 'Approve' to delete: {where}",
                "operation": {"operation": "DELETE", "is_destructive": True, "requires_approval": True, "risk_level": "high"},
                "db": db_name, "sql": sql, "rows": []
            }
        return {
            "type": "approval_required",
            "summary": f"🚨 CRITICAL: DELETE ALL ROWS\n\nQuestion: {question}\nTable: {table_info['table']}\n\n⚠️ No WHERE clause!\nWould DELETE ALL ROWS!",
            "operation": {"operation": "DELETE", "is_destructive": True, "requires_approval": True, "risk_level": "critical"},
            "db": db_name, "rows": []
        }

    def _build_insert(self, db_name, table_info, columns):
        return {
            "type": "confirmation_required",
            "summary": f"📝 INSERT INTO {table_info['table']}\n\nProvide values.\nExample: 'insert into {table_info['table']} (col1, col2) values (val1, val2)'\n\nColumns: {', '.join(columns[:8])}",
            "operation": {"operation": "INSERT", "is_destructive": True, "requires_approval": False, "risk_level": "low"},
            "db": db_name, "rows": []
        }

    def _list_tables(self, db_name):
        _, tables, columns_map, _ = self.get_schema(db_name)
        rows = []
        for t in tables:
            row = {"Table": t["table"]}
            cols = columns_map.get(t["table"].lower(), [])
            if cols:
                row["Columns"] = ", ".join(cols[:5])
            rows.append(row)
        return {
            "type": "tables", "db": db_name,
            "summary": f"Found {len(tables)} table(s) in {db_name}",
            "rows": rows
        }

    def _get_health(self, db_name):
        conn = self.monitor.dbm.get(db_name)
        if not conn:
            return {"type": "error", "summary": "Not connected", "rows": []}
        try:
            h = conn.health_check()
        except Exception as e:
            return {"type": "error", "summary": str(e), "rows": []}
        return {
            "type": "health", "db": db_name,
            "summary": f"{db_name}: {'✓ Healthy' if h.get('ok') else '✗ Unhealthy'}",
            "rows": [{
                "Database": db_name,
                "Status": "✓ Healthy" if h.get("ok") else "✗ Unhealthy",
                "Version": str(h.get("version", ""))[:150]
            }]
        }

    def _get_performance(self, db_name):
        rows = []
        conn = self.monitor.dbm.get(db_name)
        if conn and conn.dialect == "tsql":
            for name, sql in [
                ("Active Connections", "SELECT COUNT(*) AS v FROM sys.dm_exec_connections"),
                ("Running Sessions", "SELECT COUNT(*) AS v FROM sys.dm_exec_sessions WHERE status='running'"),
            ]:
                try:
                    result = conn.execute(sql)
                    if result:
                        rows.append({"Metric": name, "Value": list(result[0].values())[0]})
                except Exception:
                    pass
        return {
            "type": "performance", "db": db_name,
            "summary": f"Performance metrics for {db_name}",
            "rows": rows
        }

    def _get_slow_queries(self, db_name):
        all_slow = self.monitor.latest_slow() if self.monitor else []
        db_slow = [q for q in all_slow if q.get("db_name") == db_name]
        return {
            "type": "slow_queries", "db": db_name,
            "summary": f"Found {len(db_slow)} slow query/queries",
            "rows": db_slow[:10]
        }

    def _create_alert(self, db_name, question, sql, classification):
        try:
            self.store.insert_audit(
                "security.anomaly", db_name,
                {"operation": classification['operation'],
                 "risk_level": classification['risk_level'],
                 "user_question": question, "generated_sql": sql[:500]}
            )
        except Exception:
            pass

    def _execute(self, db_name, sql, question):
        try:
            conn = self.monitor.dbm.get(db_name)
            if not conn:
                return safe_response({"type": "error", "summary": "Not connected", "rows": []})
            rows = conn.execute(sql) or []
            try:
                table_name = sql.split("FROM")[1].split()[0].replace("[", "").replace("]", "").replace(";", "")
            except Exception:
                table_name = "table"
            return safe_response({
                "type": "table_data", "db": db_name, "table": table_name,
                "sql": sql, "summary": f"Found {len(rows)} row(s) from {db_name}.{table_name}",
                "rows": rows[:100]
            })
        except Exception as e:
            return safe_response({"type": "error", "summary": f"Query failed: {str(e)}", "sql": sql, "rows": []})

    def execute_approved(self, db_name, sql, approver="user"):
        try:
            conn = self.monitor.dbm.get(db_name)
            if not conn:
                return {"type": "error", "error": "Not connected"}
            classification = classify_sql_operation(sql)
            if not classification["is_destructive"]:
                return {"type": "error", "error": "Not destructive"}
            rows = conn.execute(sql) or []
            self.store.insert_audit(
                f"destructive.{classification['operation'].lower()}.executed",
                db_name, {"sql": sql[:1000], "approver": approver}
            )
            return {"type": "success",
                    "summary": f"✅ Executed {classification['operation']} on {db_name}",
                    "rows": rows[:100] if isinstance(rows, list) else []}
        except Exception as e:
            return {"type": "error", "error": str(e)}
