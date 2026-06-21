"""Monitor that runs the collection loop in a separate thread."""
import asyncio
import logging
import threading
import time
from datetime import datetime
from ..db.manager import DatabaseManager
from ..storage.sqlite_store import SqliteStore
from ..analyzer.security import SecurityAnalyzer
from .metrics import MetricPoint

log = logging.getLogger("aidba.monitor")


class Monitor:
    def __init__(self, cfg):
        self.cfg = cfg
        self._stop_event = threading.Event()
        self._latest_slow = {}
        self.security = SecurityAnalyzer()
        self.store = SqliteStore(cfg.app.data_dir / "aidba.db")
        self.store.init()
        self.dbm = DatabaseManager(cfg.databases or [])
        self._thread = None
        print(f"[AIDBA] Monitor initialized. DBs: {self.dbm.list()}", flush=True)

    def start_background(self):
        """Start the collection loop in a background thread."""
        if self._thread and self._thread.is_alive():
            print(f"[AIDBA] Monitor thread already running", flush=True)
            return

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[AIDBA] Monitor thread started", flush=True)

    def _run_loop(self):
        """The actual collection loop running in a background thread."""
        print(f"[AIDBA] Collection loop started", flush=True)
        
        # Do immediate first collection
        try:
            self._tick_sync()
            print(f"[AIDBA] Initial collection complete", flush=True)
        except Exception as e:
            print(f"[AIDBA] Initial collection error: {e}", flush=True)
        
        while not self._stop_event.is_set():
            # Wait for the interval
            interval = self.cfg.monitoring.critical_interval_seconds
            if self._stop_event.wait(timeout=interval):
                break
            
            # Do the collection
            try:
                self._tick_sync()
            except Exception as e:
                print(f"[AIDBA] Collection error: {e}", flush=True)
        
        print(f"[AIDBA] Collection loop stopped", flush=True)

    def _tick_sync(self):
        """Synchronous collection - called from the background thread."""
        if not self.dbm or not self.dbm.connectors:
            print(f"[AIDBA] No databases to collect from", flush=True)
            return
        
        for name, conn in self.dbm.connectors.items():
            try:
                self._collect_one(name, conn)
            except Exception as e:
                print(f"[AIDBA] Error collecting from {name}: {e}", flush=True)

    def _collect_one(self, name, conn):
        """Collect slow queries + live performance metrics."""
        # 1. Slow queries
        try:
            df = conn.collect_slow_queries(
                self.cfg.monitoring.slow_query_threshold_ms
            )
            if hasattr(df, "to_dict") and hasattr(df, "empty") and not df.empty:
                slow = df.head(50).to_dict(orient="records")
            else:
                slow = []
        except Exception:
            slow = []
        
        self._latest_slow[name] = slow
        
        for r in slow:
            try:
                self.store.insert_metric(MetricPoint(
                    db_name=name,
                    metric="query.avg_ms",
                    value=float(r.get("avg_ms") or 0),
                    labels={"query_id": str(r.get("query_id") or r.get("query_hash") or "")[:32]}
                ))
            except Exception:
                pass

        # 2. Health check
        try:
            h = conn.health_check()
            self.store.insert_metric(MetricPoint(
                db_name=name,
                metric="db.health",
                value=1.0 if h.get("ok") else 0.0,
                labels={"version": str(h.get("version", ""))[:60]}
            ))
        except Exception:
            pass

        # 3. LIVE PERFORMANCE METRICS
        perf_count = self._collect_performance_metrics(name, conn)
        
        print(f"[AIDBA] [{name}] {len(slow)} slow queries + {perf_count} perf metrics", flush=True)

    def _collect_performance_metrics(self, name, conn):
        """Collect live performance metrics."""
        count = 0
        ts = datetime.utcnow().isoformat() + "Z"
        
        try:
            if conn.dialect == "tsql":
                # SQL Server metrics
                metrics = [
                    ("active_connections", "SELECT COUNT(*) AS v FROM sys.dm_exec_connections"),
                    ("cpu_usage_pct", """
                        SELECT CAST(CAST(cpu_busy AS FLOAT) /
                            NULLIF(cpu_busy + idle, 0) * 100 AS DECIMAL(5,2)) AS v
                        FROM sys.dm_os_sys_info
                    """),
                    ("cache_hit_ratio", """
                        SELECT CAST(
                            (SELECT cntr_value FROM sys.dm_os_performance_counters
                             WHERE counter_name = 'Buffer cache hit ratio') * 1.0 /
                            NULLIF((SELECT cntr_value FROM sys.dm_os_performance_counters
                             WHERE counter_name = 'Buffer cache hit ratio base'), 0) * 100
                        AS DECIMAL(5,2)) AS v
                    """),
                    ("running_sessions", """
                        SELECT COUNT(*) AS v FROM sys.dm_exec_sessions
                        WHERE status = 'running'
                    """),
                    ("all_sessions", "SELECT COUNT(*) AS v FROM sys.dm_exec_sessions"),
                    ("query_throughput_qps", """
                        SELECT ISNULL(SUM(execution_count), 0) AS v
                        FROM sys.dm_exec_query_stats
                    """),
                    ("slow_queries_count", """
                        SELECT COUNT(*) AS v FROM sys.dm_exec_query_stats
                        WHERE total_elapsed_time / execution_count > 500000
                    """),
                    ("user_databases", "SELECT COUNT(*) AS v FROM sys.databases WHERE state = 0"),
                    ("buffer_cache_mb", """
                        SELECT CAST(COUNT(*) * 8.0 / 1024 AS DECIMAL(10,2)) AS v
                        FROM sys.dm_os_buffer_descriptors
                    """),
                ]
                
                for metric_name, sql in metrics:
                    try:
                        result = conn.execute(sql)
                        if result and len(result) > 0:
                            v = list(result[0].values())[0] if result[0] else 0
                            try:
                                v = round(float(v), 2)
                            except (TypeError, ValueError):
                                v = 0
                            self.store.insert_metric(MetricPoint(
                                db_name=name,
                                metric=metric_name,
                                value=v,
                                labels={"timestamp": ts}
                            ))
                            count += 1
                    except Exception as e:
                        print(f"[AIDBA]   {metric_name} skipped: {str(e)[:60]}", flush=True)
                        continue
                        
            elif conn.dialect == "mysql":
                metrics = [
                    ("active_connections", "SHOW GLOBAL STATUS LIKE 'Threads_connected'"),
                    ("running_threads", "SHOW GLOBAL STATUS LIKE 'Threads_running'"),
                    ("slow_queries_count", "SHOW GLOBAL STATUS LIKE 'Slow_queries'"),
                    ("uptime_seconds", "SHOW GLOBAL STATUS LIKE 'Uptime'"),
                ]
                for metric_name, sql in metrics:
                    try:
                        result = conn.execute(sql)
                        if result and len(result) > 0:
                            for r in result:
                                v = r.get("Value", 0)
                                try:
                                    v = float(v)
                                except (TypeError, ValueError):
                                    pass
                                self.store.insert_metric(MetricPoint(
                                    db_name=name,
                                    metric=metric_name,
                                    value=v,
                                    labels={"timestamp": ts}
                                ))
                                count += 1
                    except Exception:
                        continue
            else:  # postgres
                metrics = [
                    ("active_connections",
                     "SELECT count(*) AS v FROM pg_stat_activity WHERE state = 'active'"),
                    ("total_connections", "SELECT count(*) AS v FROM pg_stat_activity"),
                    ("cache_hit_ratio", """
                        SELECT ROUND(
                            100.0 * sum(blks_hit) / NULLIF(sum(blks_hit) + sum(blks_read), 0), 2
                        ) AS v FROM pg_stat_database
                    """),
                ]
                for metric_name, sql in metrics:
                    try:
                        result = conn.execute(sql)
                        if result and len(result) > 0:
                            v = list(result[0].values())[0] if result[0] else 0
                            try:
                                v = round(float(v), 2)
                            except (TypeError, ValueError):
                                v = 0
                            self.store.insert_metric(MetricPoint(
                                db_name=name,
                                metric=metric_name,
                                value=v,
                                labels={"timestamp": ts}
                            ))
                            count += 1
                    except Exception:
                        continue
        except Exception as e:
            print(f"[AIDBA] Performance metrics error: {e}", flush=True)
        
        return count

    # =================== Public APIs ===================
    
    async def stop(self):
        """Stop the monitor (signal the background thread)."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    async def run_forever(self):
        """Async stub - the actual loop runs in a thread."""
        self.start_background()
        # Keep this coroutine alive
        while not self._stop_event.is_set():
            await asyncio.sleep(1)

    def latest_slow(self, db=None):
        if db:
            return list(self._latest_slow.get(db, []))
        out = []
        for k, v in self._latest_slow.items():
            for r in v:
                rr = dict(r)
                rr["db_name"] = k
                out.append(rr)
        return out

    def get_health(self):
        try:
            return self.dbm.health_all()
        except Exception:
            return {}

    def get_db(self, name):
        return self.dbm.get(name)
