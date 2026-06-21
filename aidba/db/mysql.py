"""MySQL connector using PyMySQL - bulletproof."""
import logging
import pandas as pd
from .base import BaseConnector

log = logging.getLogger("aidba.db.mysql")

try:
    import pymysql  # type: ignore
    HAS_PYMYSQL = True
except ImportError:
    HAS_PYMYSQL = False
    pymysql = None


class MySQLConnector(BaseConnector):
    dialect = "mysql"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._conn = None

    def connect(self):
        if not HAS_PYMYSQL:
            raise RuntimeError("pymysql not installed. Run: pip install pymysql")
        
        self._conn = pymysql.connect(
            host=self.get_cfg("host", "localhost"),
            port=int(self.get_cfg("port", 3306)),
            user=self.get_cfg("username", "root"),
            password=self.get_cfg("password", ""),
            database=self.get_cfg("database", "mysql"),
            autocommit=True,
            connect_timeout=10,
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
            return cur.fetchall()
        finally:
            cur.close()

    def health_check(self):
        if not HAS_PYMYSQL:
            return {"ok": False, "error": "pymysql not installed"}
        try:
            if not self._conn:
                self.connect()
            row = self.execute("SELECT VERSION() AS v")
            return {
                "ok": True,
                "version": str(row[0].get("v", "")) if row else "unknown"
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_schemas(self):
        if not self._conn:
            return []
        try:
            cur = self._conn.cursor()
            cur.execute("SHOW DATABASES")
            return [row[0] for row in cur.fetchall()]
        except Exception as e:
            log.warning(f"get_schemas failed: {e}")
            return []

    def get_tables(self, schema):
        if not self._conn:
            return []
        try:
            cur = self._conn.cursor()
            cur.execute(f"SHOW TABLES FROM `{schema}`")
            return [row[0] for row in cur.fetchall()]
        except Exception as e:
            log.warning(f"get_tables failed: {e}")
            return []

    def collect_slow_queries(self, threshold_ms):
        if not self._conn:
            return pd.DataFrame()
        try:
            sql = (
                "SELECT digest_text AS query_text, schema_name, count_star AS exec_count, "
                f"avg_timer_wait/1e9 AS avg_ms, sum_timer_wait/1e9 AS total_ms, "
                f"sum_rows_examined/count_star AS avg_rows_examined, digest AS digest_hash "
                "FROM performance_schema.events_statements_summary_by_digest "
                f"WHERE avg_timer_wait/1e9 > {int(threshold_ms)} AND schema_name IS NOT NULL "
                "ORDER BY avg_timer_wait DESC LIMIT 50"
            )
            return pd.read_sql(sql, self._conn)
        except Exception as e:
            log.warning(f"collect_slow_queries failed: {e}")
            return pd.DataFrame()
