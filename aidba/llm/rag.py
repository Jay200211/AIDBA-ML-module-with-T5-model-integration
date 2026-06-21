"""Retrieval-Augmented Generation (RAG) for database queries.

RAG retrieves relevant context (schema, metrics, history) and injects it
into the LLM prompt so it can give better, more accurate answers.
"""
import json
import logging
import re
from typing import List, Dict, Any
from ..storage.sqlite_store import SqliteStore

log = logging.getLogger("aidba.rag")


class RAGEngine:
    """Retrieves relevant context for LLM queries."""

    def __init__(self, store: SqliteStore, monitor):
        self.store = store
        self.monitor = monitor

    def retrieve_context(self, question: str, target_db: str = None) -> Dict[str, Any]:
        """Retrieve all relevant context for a question.

        Returns a dict with:
          - schema: table/column info
          - recent_metrics: latest performance numbers
          - recent_slow_queries: recent slow queries
          - recent_audit: recent security events
          - similar_questions: past similar questions (if any)
        """
        context = {
            "schema": self._get_schema_info(target_db),
            "recent_metrics": self._get_recent_metrics(target_db),
            "recent_slow_queries": self._get_recent_slow(target_db),
            "recent_audit": self._get_recent_audit(),
            "database_health": self._get_health_summary(target_db),
        }
        return context

    def _get_schema_info(self, db_name: str = None) -> Dict[str, Any]:
        """Get table and column info for the database."""
        if not self.monitor or not self.monitor.dbm:
            return {"error": "No monitor available"}
        
        if not db_name:
            dbs = self.monitor.dbm.list()
            if not dbs:
                return {"error": "No databases connected"}
            db_name = dbs[0]
        
        conn = self.monitor.dbm.get(db_name)
        if not conn:
            return {"error": f"Database '{db_name}' not connected"}
        
        try:
            schemas = conn.get_schemas() or []
        except Exception as e:
            return {"error": f"Cannot list schemas: {e}"}
        
        tables_info = {}
        for schema in schemas[:10]:  # Limit to first 10 schemas
            try:
                tables = conn.get_tables(schema) or []
                for t in tables[:20]:  # Limit to first 20 tables per schema
                    # Get column info with a sample query
                    full_name = f"{schema}.{t}" if schema else t
                    sample_sql = self._get_sample_sql(conn.dialect, schema, t)
                    try:
                        sample = conn.execute(sample_sql) or []
                        if sample:
                            tables_info[full_name] = {
                                "columns": list(sample[0].keys()) if sample else [],
                                "row_count_sample": len(sample),
                                "sample_row": dict(list(sample[0].items())[:5]) if sample else {}
                            }
                    except Exception:
                        tables_info[full_name] = {"columns": [], "error": "Cannot sample"}
            except Exception:
                continue
        
        return {
            "database": db_name,
            "dialect": conn.dialect,
            "schemas": schemas[:10],
            "tables": tables_info,
            "total_tables": sum(len(v) if isinstance(v, list) else 1 for v in tables_info.values()) if isinstance(tables_info, dict) else 0
        }

    def _get_recent_metrics(self, db_name: str = None) -> List[Dict]:
        """Get recent performance metrics."""
        try:
            if db_name:
                rows = self.store.query_metrics(db_name=db_name, since_min=60, limit=20)
            else:
                rows = self.store.query_metrics(since_min=60, limit=20)
            return rows
        except Exception as e:
            return [{"error": str(e)}]

    def _get_recent_slow(self, db_name: str = None) -> List[Dict]:
        """Get recent slow queries."""
        if not self.monitor:
            return []
        all_slow = self.monitor.latest_slow() or []
        if db_name:
            return [q for q in all_slow if q.get("db_name") == db_name][:5]
        return all_slow[:5]

    def _get_recent_audit(self) -> List[Dict]:
        """Get recent audit events."""
        try:
            return self.store.list_audit(10)
        except Exception:
            return []

    def _get_health_summary(self, db_name: str = None) -> Dict:
        """Get database health summary."""
        if not self.monitor:
            return {}
        try:
            health = self.monitor.get_health()
            if db_name:
                return {db_name: health.get(db_name, {})}
            return health
        except Exception:
            return {}

    def _get_sample_sql(self, dialect: str, schema: str, table: str) -> str:
        full_name = f"{schema}.{table}" if schema else table
        if dialect == "tsql":
            return f"SELECT TOP 1 * FROM {full_name}"
        elif dialect == "mysql":
            return f"SELECT * FROM `{table}` LIMIT 1"
        else:
            return f"SELECT * FROM {full_name} LIMIT 1"

    def build_rag_prompt(self, question: str, target_db: str = None) -> str:
        """Build an enhanced prompt with RAG context.

        This is the magic: the LLM gets:
          1. The user's question
          2. The database schema
          3. Recent metrics
          4. Recent slow queries
          5. Recent audit events
        So it can answer intelligently.
        """
        context = self.retrieve_context(question, target_db)

        # Build the RAG prompt
        prompt = f"""You are an expert database administrator AI assistant.

USER QUESTION: {question}

"""

        # Add database schema context
        if context.get("schema") and "error" not in context["schema"]:
            schema = context["schema"]
            prompt += f"""DATABASE CONTEXT:
- Database: {schema.get('database', 'unknown')}
- Dialect: {schema.get('dialect', 'unknown')}
- Schemas: {schema.get('schemas', [])}
- Tables and their columns:
"""
            for table_name, table_info in list(schema.get("tables", {}).items())[:10]:
                cols = table_info.get("columns", [])
                prompt += f"  - {table_name}: columns = {cols[:10]}\n"

        # Add recent metrics
        metrics = context.get("recent_metrics", [])
        if metrics and len(metrics) > 0:
            prompt += f"\nRECENT METRICS (last hour):\n"
            for m in metrics[:5]:
                prompt += f"  - {m.get('db_name', '')}: {m.get('metric', '')} = {m.get('value', '')}\n"

        # Add recent slow queries
        slow = context.get("recent_slow_queries", [])
        if slow:
            prompt += f"\nRECENT SLOW QUERIES:\n"
            for q in slow[:3]:
                prompt += f"  - {q.get('db_name', '')}: {q.get('avg_ms', 0):.1f}ms avg - {str(q.get('query_text', ''))[:100]}\n"

        # Add health summary
        health = context.get("database_health", {})
        if health:
            prompt += f"\nDATABASE HEALTH:\n"
            for db, h in health.items():
                status = "OK" if h.get("ok") else "FAILED"
                prompt += f"  - {db}: {status}\n"
                if h.get("version"):
                    prompt += f"    Version: {h.get('version', '')[:100]}\n"
                if h.get("error"):
                    prompt += f"    Error: {h.get('error', '')[:100]}\n"

        # Add recent audit
        audit = context.get("recent_audit", [])
        if audit:
            prompt += f"\nRECENT AUDIT EVENTS:\n"
            for a in audit[:3]:
                prompt += f"  - {a.get('ts', '')}: {a.get('event_type', '')} ({a.get('db_name', '')})\n"

        # Final instructions
        prompt += f"""

Based on the above context, answer the user's question.
Be specific, use the data provided, and format your response clearly.
If the question asks for a list, format as a list.
If it asks for analysis, provide a brief analysis with bullet points.
"""

        return prompt

    def format_response(self, response: str, question: str) -> Dict[str, Any]:
        """Format the LLM's response into a structured output."""
        return {
            "type": "rag_response",
            "summary": response,
            "rows": [],
            "question": question
        }
