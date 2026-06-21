"""SQLite store for metrics and audit log + proposals."""
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path


class SqliteStore:
    def __init__(self, db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._path = str(db_path)
        self._lock = threading.Lock()
        self._cx = sqlite3.connect(self._path, check_same_thread=False)
        self._cx.row_factory = sqlite3.Row

    def init(self):
        """Initialize database tables."""
        with self._lock:
            self._cx.executescript("""
            CREATE TABLE IF NOT EXISTS metrics(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                db_name TEXT NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                labels TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_db_metric_ts
              ON metrics(db_name, metric, ts DESC);
            CREATE TABLE IF NOT EXISTS audit_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                db_name TEXT,
                event_type TEXT NOT NULL,
                payload TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts DESC);
            CREATE TABLE IF NOT EXISTS proposals(
                id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                db_name TEXT NOT NULL,
                title TEXT NOT NULL,
                state TEXT NOT NULL,
                payload TEXT NOT NULL,
                approver TEXT,
                comment TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_proposals_state
              ON proposals(state);
            CREATE INDEX IF NOT EXISTS idx_proposals_ts
              ON proposals(ts DESC);
            """)
            self._cx.commit()

    # =================== Metrics ===================
    def insert_metric(self, m):
        with self._lock:
            self._cx.execute(
                "INSERT INTO metrics(ts, db_name, metric, value, labels) VALUES(?,?,?,?,?)",
                (m.ts, m.db_name, m.metric, m.value, json.dumps(m.labels) if not isinstance(m.labels, str) else m.labels)
            )
            self._cx.commit()

    def query_metrics(self, db_name=None, metric=None, since_min=60, limit=1000):
        sql = "SELECT * FROM metrics WHERE 1=1"
        args = []
        if db_name:
            sql += " AND db_name=?"; args.append(db_name)
        if metric:
            sql += " AND metric=?"; args.append(metric)
        sql += " AND ts > datetime('now', ?)"
        args.append(f"-{since_min} minutes")
        sql += " ORDER BY ts DESC LIMIT ?"; args.append(limit)
        with self._lock:
            cur = self._cx.execute(sql, args)
            return [dict(r) for r in cur.fetchall()]

    # =================== Audit ===================
    def insert_audit(self, event_type, db_name, payload):
        with self._lock:
            self._cx.execute(
                "INSERT INTO audit_log(ts, db_name, event_type, payload) VALUES(?,?,?,?)",
                (datetime.utcnow().isoformat() + "Z", db_name, event_type,
                 json.dumps(payload, default=str))
            )
            self._cx.commit()

    def list_audit(self, limit=100):
        with self._lock:
            cur = self._cx.execute(
                "SELECT * FROM audit_log ORDER BY ts DESC LIMIT ?", (limit,)
            )
            return [dict(r) for r in cur.fetchall()]

    # =================== Proposals ===================
    def upsert_proposal(self, p):
        """Insert or update a proposal."""
        with self._lock:
            self._cx.execute("""
                INSERT INTO proposals(id, ts, db_name, title, state, payload, approver, comment)
                VALUES(:id, :ts, :db_name, :title, :state, :payload, :approver, :comment)
                ON CONFLICT(id) DO UPDATE SET
                  state=excluded.state,
                  approver=excluded.approver,
                  comment=excluded.comment,
                  ts=excluded.ts
            """, p)
            self._cx.commit()

    def list_proposals(self, state=None):
        """List all proposals, optionally filtered by state."""
        sql = "SELECT * FROM proposals"
        args = []
        if state:
            sql += " WHERE state=?"; args.append(state)
        sql += " ORDER BY ts DESC"
        with self._lock:
            cur = self._cx.execute(sql, args)
            return [dict(r) for r in cur.fetchall()]

    def get_proposal(self, pid):
        """Get a specific proposal by ID."""
        with self._lock:
            cur = self._cx.execute("SELECT * FROM proposals WHERE id=?", (pid,))
            r = cur.fetchone()
            return dict(r) if r else None
