"""NLQ Engine - fetches and returns data, dashboard renders as tables."""
import json
import logging
import re
from typing import Optional, List, Dict, Any
from ..llm import get_llm
from ..storage.sqlite_store import SqliteStore

log = logging.getLogger("aidba.nlq")


class NLQEngine:
    def __init__(self, cfg, store: SqliteStore, monitor):
        self.cfg = cfg
        self.store = store
        self.monitor = monitor
        try:
            self.llm = get_llm(cfg)
            self.has_llm = True
            self.provider = cfg.llm.provider
        except Exception:
            self.llm = None
            self.has_llm = False
            self.provider = "mock"

    async def ask(self, question: str) -> dict:
        q = (question or "").strip()
        if not q:
            return {"type": "error", "error": "Empty question", "rows": []}

        # Clean the question - strip punctuation, extra spaces
        original_q = q
        q = q.rstrip(".?!,").strip()  # Remove trailing punctuation
        print(f"[NLQ] Original: '{original_q}' | Cleaned: '{q}'", flush=True)

        dbs = self.monitor.dbm.list() if self.monitor and self.monitor.dbm else []
        if not dbs:
            return {
                "type": "error",
                "error": "❌ No databases connected! Add one via '+ Add Database'.",
                "rows": []
            }

        target_db = self._resolve_db(q, dbs)
        if not target_db:
            return {
                "type": "info",
                "summary": f"Please specify a database. Available: {dbs}",
                "rows": [{"Available Databases": ", ".join(dbs)}]
            }

        ql = q.lower()

        # ===== ROUTING =====

        # "list tables" / "show tables" / "what tables"
        if re.search(r"\b(list|show|what)\s+tables?\b", ql) and \
           not re.search(r"\b(list|show)\s+tables?\s+(in|from|of)\s+\w+", ql):
            return self._list_tables(target_db)

        # "list tables in MyDatabase" or "list all tables"
        if re.search(r"\b(all\s+)?tables?\b", ql) and "list" in ql:
            return self._list_tables(target_db)

        # "show customers" / "show orders" / "get users"
        if re.search(r"^(show|get|find|fetch|display|see|view)\s+(\w+)", ql):
            return await self._show_table(q, target_db)

        # "how many customers"
        if re.search(r"\bhow\s+many\b", ql):
            return await self._count_table(q, target_db)

        # "health" / "status"
        if re.search(r"\b(health|status|version)\b", ql):
            return self._get_health(target_db)

        # "performance" / "metrics"
        if re.search(r"\b(performance|metric|cpu|memory|disk|connection)\b", ql):
            return self._get_performance(target_db)

        # "slow queries"
        if re.search(r"\bslow\b.*\bquer", ql):
            return self._get_slow_queries(target_db)

        # "security" / "audit"
        if re.search(r"\b(security|audit|safe)\b", ql):
            return self._get_security_metrics()

        # "list databases"
        if re.search(r"\b(list|show|what).*databases?\b", ql):
            return self._list_databases(dbs)

        # Raw SQL
        if re.search(r"^select\s", q, re.I):
            return await self._execute_sql(q, target_db)

        # Default: try to show as table
        return await self._show_table(q, target_db)

    def _resolve_db(self, question: str, dbs: List[str]) -> Optional[str]:
        ql = question.lower()
        for db in dbs:
            if db.lower() in ql:
                return db
        if len(dbs) == 1:
            return dbs[0]
        return None

    def _list_databases(self, dbs: List[str]) -> dict:
        health = self.monitor.get_health() if self.monitor else {}
        rows = []
        for db in dbs:
            h = health.get(db, {})
            rows.append({
                "Database": db,
                "Status": "✓ Connected" if h.get("ok") else "✗ Failed",
                "Version": (h.get("version") or h.get("error", ""))[:80]
            })
        return {
            "type": "databases",
            "summary": f"Found {len(dbs)} database(s)",
            "rows": rows
        }

    def _list_tables(self, db_name: str) -> dict:
        conn = self.monitor.dbm.get(db_name)
        if not conn:
            return {"type": "error", "error": f"'{db_name}' not connected", "rows": []}

        rows = []
        try:
            if conn.dialect == "tsql":
                sql = """
                SELECT
                    s.name AS [Schema],
                    t.name AS [Table],
                    CAST(t.type_desc AS VARCHAR(20)) AS [Type],
                    ISNULL(p.rows, 0) AS [RowCount]
                FROM sys.tables t
                INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
                LEFT JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0, 1)
                WHERE t.is_ms_shipped = 0
                ORDER BY s.name, t.name
                """
                print(f"[NLQ] Listing tables for {db_name}", flush=True)
                result = conn.execute(sql)
                if result:
                    for r in result:
                        rows.append({
                            "Schema": str(r.get("Schema", "")),
                            "Table": str(r.get("Table", "")),
                            "Type": str(r.get("Type", "")),
                            "Rows": r.get("RowCount", 0)
                        })
                print(f"[NLQ] Found {len(rows)} tables", flush=True)
        except Exception as e:
            print(f"[NLQ] List tables error: {e}", flush=True)
            return {"type": "error", "error": f"Failed: {e}", "rows": []}

        if not rows:
            return {
                "type": "tables", "db": db_name,
                "summary": f"No tables found in {db_name}. Try a different database.",
                "rows": []
            }

        return {
            "type": "tables", "db": db_name,
            "summary": f"Found {len(rows)} table(s) in {db_name}",
            "rows": rows
        }

    async def _show_table(self, question: str, db_name: str) -> dict:
        """Show data from a specific table the user mentioned."""
        conn = self.monitor.dbm.get(db_name)
        if not conn:
            return {"type": "error", "error": f"'{db_name}' not connected", "rows": []}

        # Extract the table name from the question
        # Remove common prefixes
        ql = question.lower()
        
        # Get all words, skip common ones
        skip_words = {
            "show", "get", "find", "fetch", "display", "list", "view", "see",
            "me", "the", "a", "an", "from", "in", "of", "data", "rows",
            "table", "tables", "database", "all", "some", "any", "please",
            "show", "give", "want", "need", "i", "you", "can", "could",
            "would", "should", "how", "what", "which", "where", "when",
            "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "shall", "may", "might",
            "jayendra", "sqlserver", "master", "mydatabase", "salesdb",
            "database", "db", "jayendra-sqlserver", "the", "for", "with",
            "and", "or", "but", "so", "yet", "still", "just", "only",
        }
        
        words = re.findall(r"[a-z_][a-z0-9_]*", ql)
        candidates = [w for w in words if w not in skip_words and len(w) > 2]

        print(f"[NLQ] Table candidates: {candidates}", flush=True)

        if not candidates:
            return {
                "type": "info",
                "summary": "Please specify a table name (e.g., 'show customers' or 'show orders')",
                "rows": []
            }

        target_table = candidates[0]
        print(f"[NLQ] Target table: {target_table}", flush=True)

        rows = []
        found_table = None

        # Try MANY variations
        variations = [
            target_table,
            target_table.capitalize(),
            target_table.upper(),
            target_table.lower(),
            target_table + "s",
            target_table[:-1] if target_table.endswith("s") else target_table + "s",
            "tbl_" + target_table,
            target_table.replace("_", ""),
            target_table + "es",
        ]
        # Dedupe while preserving order
        seen = set()
        variations = [v for v in variations if not (v in seen or seen.add(v))]

        for var in variations:
            try:
                if conn.dialect == "tsql":
                    sql = f"SELECT TOP 10 * FROM [{var}]"
                elif conn.dialect == "mysql":
                    sql = f"SELECT * FROM `{var}` LIMIT 10"
                else:
                    sql = f'SELECT * FROM "{var}" LIMIT 10'
                
                print(f"[NLQ] Trying: {sql}", flush=True)
                result = conn.execute(sql)
                
                if result and len(result) > 0:
                    found_table = var
                    for r in result[:20]:
                        rows.append({k: str(v)[:500] for k, v in r.items()})
                    break
            except Exception as e:
                print(f"[NLQ]   Failed: {str(e)[:50]}", flush=True)
                continue

        if not found_table:
            # Also try to get a list of all tables to help the user
            try:
                list_result = self._list_tables(db_name)
                available = [r.get("Table", "") for r in list_result.get("rows", [])]
                return {
                    "type": "info",
                    "summary": f"❌ Table '{target_table}' not found in '{db_name}'.\n\nAvailable tables: {', '.join(available[:10])}\n\nTip: try 'list tables' to see all tables first.",
                    "rows": []
                }
            except Exception:
                return {
                    "type": "info",
                    "summary": f"Table '{target_table}' not found. Try 'list tables' first.",
                    "rows": []
                }

        return {
            "type": "table_data",
            "db": db_name,
            "table": found_table,
            "summary": f"Found {len(rows)} row(s) from {db_name}.{found_table}",
            "rows": rows
        }

    async def _count_table(self, question: str, db_name: str) -> dict:
        """Count rows in a table: 'how many customers'."""
        # Extract the entity
        ql = question.lower()
        for prefix in ["how many", "count", "number of"]:
            ql = ql.replace(prefix, "").strip()
        
        # Get the entity name
        words = [w for w in ql.split() if len(w) > 2 and w not in 
                 {"are", "the", "in", "of", "we", "have", "has"}]
        
        if not words:
            return {"type": "info", "summary": "What do you want to count?", "rows": []}
        
        entity = words[0]
        conn = self.monitor.dbm.get(db_name)
        if not conn:
            return {"type": "error", "error": f"'{db_name}' not connected", "rows": []}
        
        # Try variations
        for var in [entity, entity.capitalize(), entity + "s", entity + "es"]:
            try:
                if conn.dialect == "tsql":
                    sql = f"SELECT COUNT(*) AS count FROM [{var}]"
                elif conn.dialect == "mysql":
                    sql = f"SELECT COUNT(*) AS count FROM `{var}`"
                else:
                    sql = f'SELECT COUNT(*) AS count FROM "{var}"'
                result = conn.execute(sql)
                if result and len(result) > 0:
                    count = list(result[0].values())[0]
                    return {
                        "type": "count",
                        "summary": f"Table {var} has {count} rows",
                        "rows": [{"Table": var, "Row Count": count}]
                    }
            except Exception:
                continue
        
        return {
            "type": "info",
            "summary": f"Table '{entity}' not found. Try 'list tables' first.",
            "rows": []
        }

    def _get_health(self, db_name: str) -> dict:
        conn = self.monitor.dbm.get(db_name)
        if not conn:
            return {"type": "error", "error": f"'{db_name}' not connected", "rows": []}
        try:
            health = conn.health_check()
        except Exception as e:
            return {"type": "error", "error": str(e), "rows": []}

        status = "✓ Healthy" if health.get("ok") else "✗ Unhealthy"
        return {
            "type": "health", "db": db_name,
            "summary": f"{db_name}: {status}",
            "rows": [{
                "Database": db_name,
                "Status": status,
                "Version": (health.get("version") or "")[:150],
                "Driver": health.get("driver", "N/A"),
                "Auth": health.get("auth", "N/A"),
            }]
        }

    def _get_performance(self, db_name: str) -> dict:
        conn = self.monitor.dbm.get(db_name)
        if not conn:
            return {"type": "error", "error": f"'{db_name}' not connected", "rows": []}
        rows = self._fetch_performance_metrics(conn)
        return {
            "type": "performance", "db": db_name,
            "summary": f"Performance metrics for {db_name}",
            "rows": rows
        }

    def _get_slow_queries(self, db_name: str) -> dict:
        all_slow = self.monitor.latest_slow() if self.monitor else []
        db_slow = [q for q in all_slow if q.get("db_name") == db_name]
        top = sorted(db_slow, key=lambda r: float(r.get("avg_ms") or 0), reverse=True)[:10]

        rows = [{
            "Avg ms": round(float(q.get("avg_ms", 0)), 1),
            "Exec Count": q.get("exec_count", 0),
            "Total ms": round(float(q.get("total_ms", 0)), 1),
            "Query": str(q.get("query_text", ""))[:300]
        } for q in top]

        return {
            "type": "slow_queries", "db": db_name,
            "summary": f"Found {len(db_slow)} slow query/queries in {db_name}",
            "rows": rows
        }

    def _get_security_metrics(self) -> dict:
        try:
            rows = self.store.list_audit(50)
        except Exception:
            rows = []
        security_events = [r for r in rows if "security" in str(r.get("event_type", "")).lower()]

        formatted = [{
            "Timestamp": r.get("ts", ""),
            "Database": r.get("db_name", ""),
            "Event": r.get("event_type", ""),
            "Details": str(r.get("payload", ""))[:200]
        } for r in security_events[:20]]

        return {
            "type": "security",
            "summary": f"Found {len(security_events)} security event(s)",
            "rows": formatted
        }

    def _fetch_performance_metrics(self, conn) -> List[Dict[str, Any]]:
        rows = []
        try:
            if conn.dialect == "tsql":
                queries = [
                    ("Active Connections", "SELECT COUNT(*) AS v FROM sys.dm_exec_connections", "count"),
                    ("Running Sessions", "SELECT COUNT(*) AS v FROM sys.dm_exec_sessions WHERE status='running'", "count"),
                    ("All Sessions", "SELECT COUNT(*) AS v FROM sys.dm_exec_sessions", "count"),
                    ("User Databases", "SELECT COUNT(*) AS v FROM sys.databases WHERE state=0", "count"),
                    ("CPU Usage (%)", """
                        SELECT CAST(CAST(cpu_busy AS FLOAT) /
                            NULLIF(cpu_busy + idle, 0) * 100 AS DECIMAL(5,2)) AS v
                        FROM sys.dm_os_sys_info
                    """, "percent"),
                ]
                for name, sql, unit in queries:
                    try:
                        result = conn.execute(sql)
                        if result and len(result) > 0:
                            v = list(result[0].values())[0] if result[0] else 0
                            try: v = round(float(v), 2)
                            except (TypeError, ValueError): v = 0
                            rows.append({"Metric": name, "Value": v, "Unit": unit})
                    except Exception:
                        pass
            elif conn.dialect == "mysql":
                for name, sql in [
                    ("Active Connections", "SHOW GLOBAL STATUS LIKE 'Threads_connected'"),
                    ("Running Threads", "SHOW GLOBAL STATUS LIKE 'Threads_running'"),
                ]:
                    try:
                        result = conn.execute(sql)
                        if result:
                            for r in result:
                                rows.append({
                                    "Metric": str(r.get("Variable_name", name)),
                                    "Value": r.get("Value", 0),
                                    "Unit": "count"
                                })
                    except Exception:
                        pass
            else:  # postgres
                sql = "SELECT count(*) AS total, count(*) FILTER (WHERE state='active') AS active FROM pg_stat_activity"
                try:
                    result = conn.execute(sql)
                    if result:
                        r = result[0]
                        rows.append({"Metric": "Total Connections", "Value": r.get("total", 0), "Unit": "count"})
                        rows.append({"Metric": "Active Connections", "Value": r.get("active", 0), "Unit": "count"})
                except Exception:
                    pass
        except Exception:
            pass

        if not rows:
            rows.append({"Metric": "info", "Value": "No metrics", "Unit": "info"})
        return rows

    async def _execute_sql(self, sql: str, db_name: str) -> dict:
        conn = self.monitor.dbm.get(db_name)
        if not conn:
            return {"type": "error", "error": f"'{db_name}' not connected", "rows": []}
        try:
            rows = conn.execute(sql) or []
            return {
                "type": "query", "db": db_name, "sql": sql,
                "summary": f"Returned {len(rows)} row(s)",
                "rows": rows[:100]
            }
        except Exception as e:
            return {"type": "error", "error": f"Query failed: {e}", "sql": sql, "rows": []}
