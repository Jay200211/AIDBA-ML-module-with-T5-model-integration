"""PostgreSQL connector using pg8000 - bulletproof."""
import logging
import pandas as pd
from .base import BaseConnector

log = logging.getLogger("aidba.db.postgres")

try:
    import pg8000  # type: ignore
    HAS_PG8000 = True
except ImportError:
    HAS_PG8000 = False
    pg8000 = None


class PostgresConnector(BaseConnector):
    dialect = "postgres"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._conn = None

    def connect(self):
        if not HAS_PG8000:
            raise RuntimeError("pg8000 not installed. Run: pip install pg8000")
        
        self._conn = pg8000.connect(
            host=self.get_cfg("host", "localhost"),
            port=int(self.get_cfg("port", 5432)),
            database=self.get_cfg("database", "postgres"),
            user=self.get_cfg("username", "postgres"),
            password=self.get_cfg("password", ""),
            timeout=10,
        )
        return self

    def disconnect(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def execute(self, sql, params=None):
        if not self._conn:
            raise RuntimeError("Not connected")
        cur = self._conn.cursor()
        try:
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            if cur.description:
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                return rows
            return []
        finally:
            cur.close()

    def health_check(self):
        if not HAS_PG8000:
            return {"ok": False, "error": "pg8000 not installed"}
        try:
            if not self._conn:
                self.connect()
            row = self.execute("SELECT version()")
            return {
                "ok": True,
                "version": str(row[0].get("version", ""))[:100] if row else "unknown"
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_schemas(self):
        if not self._conn:
            return []
        try:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT nspname FROM pg_namespace "
                "WHERE nspname NOT LIKE 'pg_%' AND nspname <> 'information_schema' "
                "ORDER BY nspname"
            )
            return [row[0] for row in cur.fetchall()]
        except Exception as e:
            log.warning(f"get_schemas failed: {e}")
            return []

    def get_tables(self, schema):
        if not self._conn:
            return []
        try:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = %s ORDER BY tablename",
                (schema,)
            )
            return [row[0] for row in cur.fetchall()]
        except Exception as e:
            log.warning(f"get_tables failed: {e}")
            return []

    def collect_slow_queries(self, threshold_ms):
        if not self._conn:
            return pd.DataFrame()
        try:
            sql = (
                "SELECT query AS query_text, calls AS exec_count, "
                "mean_exec_time AS avg_ms, total_exec_time AS total_ms, "
                "rows AS avg_rows, CAST(queryid AS TEXT) AS query_id "
                "FROM pg_stat_statements "
                f"WHERE mean_exec_time > {int(threshold_ms)} "
                "ORDER BY mean_exec_time DESC LIMIT 50"
            )
            return pd.read_sql(sql, self._conn)
        except Exception as e:
            log.warning(f"collect_slow_queries failed: {e}")
            return pd.DataFrame()
