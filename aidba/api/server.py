"""AIDBA FastAPI server - complete version with all endpoints.

Includes:
- Health checks
- Database management
- Slow query detection
- Performance metrics
- NLQ with ML integration (synchronous call - no await)
- Approval workflow
- Excel-compatible CSV exports
- SSE for live updates
"""
import asyncio
import csv
import io
import json
import logging
import threading
import yaml
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

from ..db.manager import DatabaseManager
from ..monitor.collector import Monitor
from ..storage.sqlite_store import SqliteStore
from ..approval.workflow import ApprovalWorkflow

log = logging.getLogger("aidba.api")

_monitor = None
_workflow = None
_config = None
_nlq_engine = None


def build_app(cfg):
    """Build the FastAPI app with all endpoints."""
    global _monitor, _workflow, _config

    _config = cfg
    app = FastAPI(title="AIDBA", version="0.1.0")

    # Initialize SQLite store
    store = SqliteStore(cfg.app.data_dir / "aidba.db")
    store.init()
    print(f"[AIDBA] SQLite store initialized", flush=True)

    # Initialize monitor with database connections
    try:
        _monitor = Monitor(cfg)
        # Start the monitor in a background thread
        def run_monitor():
            try:
                asyncio.run(_monitor.run_forever())
            except Exception as e:
                print(f"[AIDBA] Monitor thread error: {e}", flush=True)

        monitor_thread = threading.Thread(target=run_monitor, daemon=True)
        monitor_thread.start()
        print(f"[AIDBA] Monitor thread started", flush=True)
    except Exception as e:
        print(f"[AIDBA] Monitor init failed: {e}", flush=True)
        _monitor = None

    # Initialize approval workflow
    _workflow = ApprovalWorkflow(store)
    print(f"[AIDBA] Approval workflow initialized", flush=True)

    # Mount static files
    static_dir = Path(__file__).parent.parent / "dashboard"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # =================== Dashboard ===================
    @app.get("/", response_class=HTMLResponse)
    async def root():
        idx = static_dir / "index.html"
        if idx.exists():
            return HTMLResponse(idx.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>AIDBA</h1><p>Dashboard not built.</p>")

    # =================== Health ===================
    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "databases": _monitor.get_health() if _monitor else {}
        }

    # =================== Databases ===================
    @app.get("/api/dbs")
    async def list_dbs():
        if not _monitor:
            return {"databases": [], "health": {}}
        return {
            "databases": _monitor.dbm.list() if _monitor.dbm else [],
            "health": _monitor.get_health()
        }

    # =================== Slow Queries ===================
    @app.get("/api/slow_queries")
    async def slow_queries(db=None, limit: int = 50):
        if not _monitor:
            return {"rows": []}
        try:
            limit = int(limit) if limit else 50
            if limit < 1:
                limit = 50
        except (ValueError, TypeError):
            limit = 50
        return {"rows": _monitor.latest_slow(db)[:limit]}

    # =================== Metrics ===================
    @app.get("/api/metrics")
    async def metrics(db: str = None, metric: str = None, since_min: int = 60):
        """Get all metrics with timestamps."""
        if not _monitor:
            return {"rows": []}
        try:
            since_min = int(since_min) if since_min else 60
            if since_min < 1:
                since_min = 60
        except (ValueError, TypeError):
            since_min = 60
        rows = _monitor.store.query_metrics(
            db_name=db, metric=metric, since_min=since_min, limit=500
        )
        return {"rows": rows}

    # =================== Proposals ===================
    @app.get("/api/proposals")
    async def list_proposals(state=None):
        if not _workflow:
            return {"proposals": []}
        try:
            rows = _workflow.list_proposals(state=state)
            for r in rows:
                try:
                    r["payload"] = json.loads(r["payload"])
                except Exception:
                    pass
            return {"proposals": rows}
        except Exception as e:
            log.exception(f"Failed to list proposals: {e}")
            return {"proposals": []}

    @app.post("/api/proposals/create_test")
    async def create_test_proposal():
        """Create a test proposal to verify the state machine works."""
        if not _workflow:
            raise HTTPException(503, "Workflow not initialized")
        try:
            pid = _workflow.create_test_proposal()
            return {"ok": True, "id": pid, "message": "Test proposal created"}
        except Exception as e:
            log.exception(f"Failed to create test proposal: {e}")
            raise HTTPException(500, str(e))

    @app.post("/api/proposals/{pid}/transition")
    async def transition_proposal(pid: str, body: dict = Body(default={})):
        """Transition a proposal to a new state."""
        if not _workflow:
            raise HTTPException(503, "Workflow not initialized")
        try:
            result = _workflow.transition(
                pid,
                body.get("state"),
                body.get("approver", "dashboard"),
                body.get("comment")
            )
            return result
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            log.exception(f"Unexpected error in transition: {e}")
            raise HTTPException(500, f"Internal error: {str(e)}")

    @app.get("/api/proposals/{pid}/history")
    async def proposal_history(pid: str):
        """Get the transition history for a proposal."""
        if not _workflow:
            raise HTTPException(503, "Workflow not initialized")
        try:
            history = _workflow.get_history(pid)
            return {"proposal_id": pid, "history": history}
        except Exception as e:
            log.exception(f"Failed to get history: {e}")
            raise HTTPException(500, str(e))

    @app.get("/api/proposals/{pid}/allowed")
    async def allowed_transitions(pid: str):
        """Get allowed next states for a proposal."""
        if not _workflow:
            raise HTTPException(503, "Workflow not initialized")
        try:
            allowed = _workflow.get_allowed_transitions(pid)
            proposal = _workflow.get_proposal(pid)
            return {
                "proposal_id": pid,
                "current_state": proposal.get("state") if proposal else None,
                "allowed_transitions": allowed
            }
        except Exception as e:
            log.exception(f"Failed to get allowed transitions: {e}")
            raise HTTPException(500, str(e))

    # =================== Audit ===================
    @app.get("/api/audit")
    async def audit(limit: int = 100):
        if not _monitor:
            return {"rows": []}
        try:
            limit = int(limit) if limit else 100
            if limit < 1:
                limit = 100
        except (ValueError, TypeError):
            limit = 100
        return {"rows": _monitor.store.list_audit(limit)}

    # =================== NLQ (FIXED - SYNCHRONOUS) ===================
    @app.post("/api/nlq")
    async def nlq(body: dict = Body(default={})):
        """LLM-driven natural language query.

        FIX: Calls _nlq_engine.ask() SYNCHRONOUSLY (no await).
        The HybridNLQEngine.ask() method is synchronous and returns a dict.
        """
        if not _monitor:
            return {
                "type": "error",
                "error": "Monitor not initialized",
                "summary": "Monitor not initialized. Check server logs.",
                "rows": []
            }

        question = body.get("question", "").strip()
        if not question:
            return {
                "type": "error",
                "error": "Empty question",
                "summary": "Please ask a question.",
                "rows": []
            }

        # Lazy-init the NLQ engine
        global _nlq_engine
        if _nlq_engine is None:
            try:
                from ..ml.integration import HybridNLQEngine
                from ..ml.predict import SQLPredictor

                # Try to load T5 model
                model_path = "D:\\aidba\\models\\aidba-sql-t5"
                predictor = None
                try:
                    predictor = SQLPredictor(model_path=model_path)
                    if predictor.loaded:
                        print(f"[AIDBA] T5 model loaded from {model_path}", flush=True)
                    else:
                        print(f"[AIDBA] T5 model not loaded, using rule-based only", flush=True)
                        predictor = None
                except Exception as e:
                    print(f"[AIDBA] T5 model not available: {e}", flush=True)
                    predictor = None

                _nlq_engine = HybridNLQEngine(_config, _monitor.store, _monitor, predictor)
                print(f"[AIDBA] Hybrid NLQ Engine initialized", flush=True)
            except Exception as e:
                log.exception(f"Failed to initialize NLQ engine: {e}")
                return {
                    "type": "error",
                    "error": f"NLQ engine init failed: {str(e)}",
                    "summary": "Could not initialize the NLQ engine.",
                    "rows": []
                }

        if not _nlq_engine:
            return {
                "type": "error",
                "error": "NLQ engine not available",
                "summary": "NLQ engine could not be initialized.",
                "rows": []
            }

        # CRITICAL FIX: Call SYNCHRONOUSLY (no await!)
        try:
            result = _nlq_engine.ask(question)
            # Ensure result is a valid dict
            if not isinstance(result, dict):
                return {
                    "type": "error",
                    "error": "Invalid response from NLQ engine",
                    "summary": "NLQ engine returned invalid response.",
                    "rows": []
                }
            return result
        except Exception as e:
            log.exception(f"NLQ error: {e}")
            return {
                "type": "error",
                "error": str(e),
                "summary": f"Error processing question: {str(e)}",
                "rows": []
            }

    @app.post("/api/nlq/execute_approved")
    async def execute_approved_sql(body: dict = Body(...)):
        """Execute an approved destructive SQL operation."""
        if not _nlq_engine:
            raise HTTPException(503, "NLQ engine not initialized")

        db_name = body.get("db", "")
        sql = body.get("sql", "")
        approver = body.get("approver", "dashboard_user")

        if not db_name or not sql:
            raise HTTPException(400, "db and sql are required")

        try:
            from ..ml.integration import HybridNLQEngine
            if isinstance(_nlq_engine, HybridNLQEngine):
                result = _nlq_engine.execute_approved(db_name, sql, approver)
            else:
                result = {"type": "error", "error": "ML engine not active"}
            return result
        except Exception as e:
            log.exception(f"Execute approved failed: {e}")
            raise HTTPException(500, str(e))

    # =================== Exports (CSV) ===================
    @app.get("/api/export/metrics")
    async def export_metrics(db: str = None, hours: int = 24):
        """Export all metrics with timestamps as CSV."""
        if not _monitor:
            raise HTTPException(503, "No monitor")
        try:
            rows = _monitor.store.query_metrics(
                db_name=db, metric=None, since_min=hours * 60, limit=50000
            )
        except Exception as e:
            raise HTTPException(500, str(e))

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Timestamp", "Database", "Metric", "Value", "Labels"])
        for r in rows:
            writer.writerow([
                r.get("ts", ""),
                r.get("db_name", ""),
                r.get("metric", ""),
                r.get("value", ""),
                r.get("labels", "")
            ])

        csv_content = output.getvalue()
        filename = f"aidba_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    @app.get("/api/export/slow_queries")
    async def export_slow_queries(db: str = None):
        """Export slow queries with timestamps."""
        if not _monitor:
            raise HTTPException(503, "No monitor")
        rows = _monitor.latest_slow(db)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Export Time", "Database", "Query Text", "Avg ms",
            "Exec Count", "Total ms", "Query Hash"
        ])
        export_time = datetime.now().isoformat()
        for r in rows:
            writer.writerow([
                export_time,
                r.get("db_name", ""),
                str(r.get("query_text", ""))[:500],
                r.get("avg_ms", 0),
                r.get("exec_count", 0),
                r.get("total_ms", 0),
                str(r.get("query_id") or r.get("query_hash") or "")[:32]
            ])

        csv_content = output.getvalue()
        filename = f"aidba_slow_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    @app.get("/api/export/audit")
    async def export_audit(hours: int = 24):
        """Export audit log with timestamps."""
        if not _monitor:
            raise HTTPException(503, "No monitor")
        try:
            rows = _monitor.store.list_audit(50000)
        except Exception:
            rows = []

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Timestamp", "Database", "Event Type", "Payload"])
        for r in rows:
            writer.writerow([
                r.get("ts", ""),
                r.get("db_name", ""),
                r.get("event_type", ""),
                str(r.get("payload", ""))[:500]
            ])

        csv_content = output.getvalue()
        filename = f"aidba_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    @app.get("/api/export/performance")
    async def export_performance():
        """Export real-time performance metrics for all databases."""
        if not _monitor:
            raise HTTPException(503, "No monitor")

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Export Time", "Database", "Metric", "Value", "Unit"])

        export_time = datetime.now().isoformat()
        try:
            from ..ml.integration import HybridNLQEngine
            nlq = HybridNLQEngine(_config, _monitor.store, _monitor, None)
            for db_name in _monitor.dbm.list():
                try:
                    conn = _monitor.dbm.get(db_name)
                    if conn:
                        metrics = nlq._fetch_performance_metrics(conn)
                        for m in metrics:
                            writer.writerow([
                                export_time,
                                db_name,
                                m.get("Metric", ""),
                                m.get("Value", ""),
                                m.get("Unit", "")
                            ])
                except Exception:
                    continue
        except Exception:
            pass

        csv_content = output.getvalue()
        filename = f"aidba_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    # =================== Database Management ===================
    @app.post("/api/test_connection")
    async def test_connection(body: dict = Body(...)):
        try:
            db_type = body.get("type", "").lower()
            host = body.get("host", "localhost")
            port = int(body.get("port", 0))
            database = body.get("database", "")
            username = body.get("username", "").strip()
            password = body.get("password", "").strip()

            if db_type == "sqlserver":
                try:
                    import pyodbc
                except ImportError:
                    return {"ok": False, "error": "pyodbc not installed. Run: pip install pyodbc"}

                available = pyodbc.drivers()
                driver = None
                for preferred in [
                    "ODBC Driver 18 for SQL Server",
                    "ODBC Driver 17 for SQL Server",
                    "SQL Server Native Client 11.0",
                    "SQL Server",
                ]:
                    if preferred in available:
                        driver = preferred
                        break
                if not driver:
                    sql_drivers = [d for d in available if "SQL Server" in d]
                    if sql_drivers:
                        driver = sql_drivers[0]

                if not driver:
                    return {"ok": False, "error": f"No SQL Server ODBC driver. Available: {available}"}

                if "\\" in host or "/" in host:
                    server = host
                else:
                    server = f"{host},{port}"

                if username and password:
                    cs = (
                        f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
                        f"UID={username};PWD={password};"
                    )
                else:
                    cs = (
                        f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
                        f"Trusted_Connection=yes;"
                    )

                if "18" in driver or "17" in driver:
                    cs += "TrustServerCertificate=yes;"
                cs += "Connection Timeout=10;"

                try:
                    conn = pyodbc.connect(cs, timeout=10, autocommit=True)
                    cur = conn.cursor()
                    cur.execute("SELECT @@VERSION AS v, DB_NAME() AS db")
                    row = cur.fetchone()
                    conn.close()
                    return {
                        "ok": True,
                        "version": str(row[0])[:100],
                        "current_db": str(row[1]) if len(row) > 1 else database,
                        "driver": driver,
                        "auth": "SQL Auth" if username else "Windows Auth"
                    }
                except Exception as e:
                    return {"ok": False, "error": str(e)}
            else:
                return {"ok": False, "error": f"Unsupported type: {db_type}"}
        except Exception as e:
            return {"ok": False, "error": f"Server error: {str(e)}"}

    @app.post("/api/add_database")
    async def add_database(body: dict = Body(...)):
        try:
            cfg_path = Path("config.yaml")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f) or {}

            if "databases" not in cfg_data or cfg_data["databases"] is None:
                cfg_data["databases"] = []

            for db in cfg_data["databases"]:
                if db.get("name") == body.get("name"):
                    return {
                        "ok": False,
                        "error": f"Database '{body.get('name')}' already exists. Use a different unique name."
                    }

            cfg_data["databases"].append({
                "name": body.get("name"),
                "type": body.get("type"),
                "host": body.get("host"),
                "port": int(body.get("port", 0)),
                "database": body.get("database"),
                "username": body.get("username", ""),
                "password": body.get("password", ""),
                "enabled": True
            })

            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg_data, f, default_flow_style=False, sort_keys=False)

            return {
                "ok": True,
                "message": f"Database '{body.get('name')}' added! Restart AIDBA to connect."
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # =================== SSE Stream ===================
    @app.get("/api/stream")
    async def stream():
        async def event_generator():
            while True:
                try:
                    payload = {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "health": _monitor.get_health() if _monitor else {},
                        "slow_count": {}
                    }
                    if _monitor and hasattr(_monitor, "_latest_slow"):
                        try:
                            payload["slow_count"] = {
                                db: len(v)
                                for db, v in _monitor._latest_slow.items()
                            }
                        except Exception:
                            pass
                    yield f"event: tick\ndata: {json.dumps(payload, default=str)}\n\n"
                except Exception:
                    pass
                await asyncio.sleep(5)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return app
