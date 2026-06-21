"""Configuration loader - handles dict/object properly."""
from pathlib import Path
from typing import Literal
import yaml


class DatabaseConfig:
    """Simple database config class."""

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'unnamed')
        self.type = kwargs.get('type', 'sqlserver')
        self.host = kwargs.get('host', 'localhost')
        self.port = kwargs.get('port', 1433)
        self.database = kwargs.get('database', 'master')
        self.username = kwargs.get('username', '')
        self.password = kwargs.get('password', '')
        self.enabled = kwargs.get('enabled', True)

    def __repr__(self):
        return f"DatabaseConfig(name={self.name}, type={self.type}, host={self.host})"


class AppConfig:
    def __init__(self, **kwargs):
        self.host = kwargs.get('host', '127.0.0.1')
        self.port = kwargs.get('port', 8000)
        self.log_level = kwargs.get('log_level', 'INFO')
        self.data_dir = Path(kwargs.get('data_dir', './data'))


class MonitoringConfig:
    def __init__(self, **kwargs):
        self.critical_interval_seconds = kwargs.get('critical_interval_seconds', 10)
        self.secondary_interval_seconds = kwargs.get('secondary_interval_seconds', 60)
        self.slow_query_threshold_ms = kwargs.get('slow_query_threshold_ms', 500)
        self.max_overhead_percent = kwargs.get('max_overhead_percent', 2.0)


class LLMConfig:
    def __init__(self, **kwargs):
        self.provider = kwargs.get('provider', 'mock')
        self.base_url = kwargs.get('base_url', 'http://localhost:11434')
        self.model = kwargs.get('model', 'deepseek-coder:6.7b')
        self.timeout_seconds = kwargs.get('timeout_seconds', 120)
        self.temperature = kwargs.get('temperature', 0.1)


class ApprovalConfig:
    def __init__(self, **kwargs):
        self.auto_approve_safe = kwargs.get('auto_approve_safe', False)
        self.rollback_on_p99_regression_pct = kwargs.get('rollback_on_p99_regression_pct', 10.0)


class StorageConfig:
    def __init__(self, **kwargs):
        self.metrics_retention_days = kwargs.get('metrics_retention_days', 7)
        self.audit_retention_days = kwargs.get('audit_retention_days', 365)


class RootConfig:
    """Root configuration container."""

    def __init__(self, **kwargs):
        self.app = AppConfig(**(kwargs.get('app') or {}))
        self.monitoring = MonitoringConfig(**(kwargs.get('monitoring') or {}))
        self.llm = LLMConfig(**(kwargs.get('llm') or {}))
        self.approval = ApprovalConfig(**(kwargs.get('approval') or {}))
        self.storage = StorageConfig(**(kwargs.get('storage') or {}))

        # Parse databases list
        self.databases = []
        for db in (kwargs.get('databases') or []):
            if isinstance(db, dict):
                self.databases.append(DatabaseConfig(**db))
            elif isinstance(db, DatabaseConfig):
                self.databases.append(db)
            else:
                # Try to convert
                try:
                    self.databases.append(DatabaseConfig(**dict(db)))
                except Exception:
                    print(f"[AIDBA] Skipping invalid DB config: {db}")


def load_config(path: str = "config.yaml"):
    """Load config.yaml with robust error handling."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw:
        print("[AIDBA] WARNING: Config is empty, using defaults", flush=True)
        raw = {}

    # Convert data_dir to Path
    if "app" in raw and isinstance(raw["app"], dict):
        raw["app"]["data_dir"] = Path(raw["app"].get("data_dir", "./data"))

    # Pre-process databases to ensure all have 'name' field
    if "databases" in raw and isinstance(raw["databases"], list):
        for i, db in enumerate(raw["databases"]):
            if not isinstance(db, dict):
                print(f"[AIDBA] WARNING: Removing non-dict database at index {i}", flush=True)
                raw["databases"][i] = None
                continue

            # Auto-generate name if missing
            if "name" not in db or not db["name"]:
                db_type = db.get("type", "db")
                host = db.get("host", "localhost").replace("\\", "_").replace("/", "_")
                port = db.get("port", 1433)
                db["name"] = f"{db_type}_{host}_{port}"
                print(f"[AIDBA] Auto-generated name for DB #{i}: {db['name']}", flush=True)

            # Ensure type is valid
            if "type" not in db:
                db["type"] = "sqlserver"
            db["type"] = str(db["type"]).lower()

            # Ensure port
            if "port" not in db:
                if db["type"] == "sqlserver":
                    db["port"] = 1433
                elif db["type"] == "mysql":
                    db["port"] = 3306
                else:
                    db["port"] = 5432

            # Ensure database name
            if "database" not in db:
                db["database"] = "master"

            # Ensure username/password
            db.setdefault("username", "")
            db.setdefault("password", "")
            db.setdefault("enabled", True)

        # Remove None entries
        raw["databases"] = [db for db in raw["databases"] if db is not None]
    else:
        raw["databases"] = []

    # Create config
    try:
        config = RootConfig(**raw)
    except Exception as e:
        print(f"[AIDBA] ERROR creating config: {e}", flush=True)
        # Fallback: try with defaults
        try:
            config = RootConfig()
            print("[AIDBA] Using default config", flush=True)
        except Exception as e2:
            print(f"[AIDBA] FATAL: {e2}", flush=True)
            raise

    print(f"[AIDBA] Config loaded: {len(config.databases)} database(s) configured", flush=True)
    for db in config.databases:
        # Safe attribute access
        name = getattr(db, 'name', 'unknown')
        db_type = getattr(db, 'type', 'unknown')
        host = getattr(db, 'host', 'unknown')
        port = getattr(db, 'port', 0)
        database = getattr(db, 'database', 'unknown')
        print(f"[AIDBA]   - {name} ({db_type}) at {host}:{port}/{database}", flush=True)

    return config
