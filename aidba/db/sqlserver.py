"""SQL Server connector using pyodbc - accepts both object and dict configs."""
import logging
import pandas as pd
from .base import BaseConnector

log = logging.getLogger("aidba.db.sqlserver")

try:
    import pyodbc
    HAS_PYODBC = True
except ImportError:
    HAS_PYODBC = False
    pyodbc = None


def _find_driver():
    """Find the best available SQL Server ODBC driver."""
    if not HAS_PYODBC:
        return None
    available = pyodbc.drivers()
    for preferred in [
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server",
        "SQL Server Native Client 11.0",
        "SQL Server",
    ]:
        if preferred in available:
            return preferred
    sql_drivers = [d for d in available if "SQL Server" in d]
    if sql_drivers:
        return sql_drivers[0]
    return None


def _build_connection_string(host, port, database, username, password, driver):
    """Build a pyodbc connection string - handles named instances."""
    if "\\" in host or "/" in host:
        server = host
    else:
        server = f"{host},{port}"

    if username and password:
        cs = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};PWD={password};"
        )
    else:
        cs = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"Trusted_Connection=yes;"
        )

    if "18" in driver or "17" in driver:
        cs += "TrustServerCertificate=yes;"
    cs += "Connection Timeout=10;"
    return cs


class SqlServerConnector(BaseConnector):
    dialect = "tsql"

    def __init__(self, cfg):
        # Handle both dict and object configs
        if isinstance(cfg, dict):
            self.cfg = type('Config', (), cfg)()  # Convert dict to object
        else:
            self.cfg = cfg
        self._driver = None
        self._conn = None

    def connect(self):
        if not HAS_PYODBC:
            raise RuntimeError("pyodbc not installed")
        self._driver = _find_driver()
        if not self._driver:
            raise RuntimeError("No SQL Server ODBC driver found")

        # Access as attributes (works for both dict-converted and object)
        cs = _build_connection_string(
            self.cfg.host, self.cfg.port, self.cfg.database,
            self.cfg.username, self.cfg.password, self._driver
        )
        log.info(f"Connecting to SQL Server: {self.cfg.host}:{self.cfg.port}/{self.cfg.database}")
        print(f"[AIDBA] SQL Server: connecting to {self.cfg.host}...", flush=True)
        try:
            self._conn = pyodbc.connect(cs, autocommit=True, timeout=10)
            print(f"[AIDBA] SQL Server: connected to {self.cfg.host}", flush=True)
            return self
        except Exception as e:
            print(f"[AIDBA] SQL Server connection failed: {e}", flush=True)
            raise

    def disconnect(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def execute(self, sql, params=None):
        if not self._conn:
            return []
        try:
            cursor = self._conn.cursor()
            try:
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                if cursor.description:
                    cols = [c[0] for c in cursor.description]
                    return [dict(zip(cols, row)) for row in cursor.fetchall()]
                return []
            finally:
                cursor.close()
        except Exception as e:
            log.warning(f"Execute failed: {e}")
            return []

    def health_check(self):
        if not self._conn:
            return {"ok": False, "error": "Not connected"}
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT @@VERSION AS v, DB_NAME() AS db")
            row = cursor.fetchone()
            cursor.close()
            return {
                "ok": True,
                "version": str(row[0])[:150] if row else "unknown",
                "driver": self._driver or "unknown",
                "auth": "SQL Auth" if self.cfg.username else "Windows Auth"
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_schemas(self):
        if not self._conn:
            return []
        try:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT name FROM sys.schemas
                WHERE name NOT IN ('sys', 'INFORMATION_SCHEMA', 'guest')
                ORDER BY name
            """)
            schemas = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return schemas
        except Exception as e:
            log.warning(f"get_schemas failed: {e}")
            return ["dbo"]

    def get_tables(self, schema=None):
        if not self._conn:
            return []
        try:
            cursor = self._conn.cursor()
            if schema and schema != "dbo":
                cursor.execute("""
                    SELECT t.name
                    FROM sys.tables t
                    INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
                    WHERE s.name = ? AND t.is_ms_shipped = 0
                    ORDER BY t.name
                """, (schema,))
            else:
                cursor.execute("""
                    SELECT t.name
                    FROM sys.tables t
                    INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
                    WHERE s.name = 'dbo' AND t.is_ms_shipped = 0
                    ORDER BY t.name
                """)
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return tables
        except Exception as e:
            log.warning(f"get_tables failed: {e}")
            return []

    def collect_slow_queries(self, threshold_ms):
        if not self._conn:
            return None
        try:
            sql = f"""
                SELECT TOP 50
                    qs.execution_count AS exec_count,
                    qs.total_elapsed_time/1000 AS total_ms,
                    (qs.total_elapsed_time/NULLIF(qs.execution_count, 0))/1000 AS avg_ms,
                    qs.total_worker_time/qs.execution_count/1000 AS avg_cpu_ms,
                    qs.total_logical_reads/qs.execution_count AS avg_logical_reads,
                    SUBSTRING(qt.text, 1, 4000) AS query_text,
                    CAST(qs.query_hash AS VARCHAR(64)) AS query_hash,
                    DB_NAME(qt.dbid) AS database_name
                FROM sys.dm_exec_query_stats qs
                CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) qt
                WHERE (qs.total_elapsed_time/NULLIF(qs.execution_count, 0))/1000 > {int(threshold_ms)}
                ORDER BY avg_ms DESC
            """
            return pd.read_sql(sql, self._conn)
        except Exception as e:
            log.warning(f"Slow query collection failed: {e}")
            return None

    def explain_query(self, sql):
        if not self._conn:
            return ""
        try:
            cursor = self._conn.cursor()
            cursor.execute("SET SHOWPLAN_ALL ON")
            try:
                cursor.execute(sql)
                rows = cursor.fetchall()
                cursor.execute("SET SHOWPLAN_ALL OFF")
                cursor.close()
                return "\n".join(str(row) for row in rows)
            except Exception:
                cursor.execute("SET SHOWPLAN_ALL OFF")
                cursor.close()
                return ""
        except Exception as e:
            return f"Error: {e}"
