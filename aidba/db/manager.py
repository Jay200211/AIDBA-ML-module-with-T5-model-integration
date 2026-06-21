"""Database manager - handles both dicts and config objects."""
import logging
import traceback

log = logging.getLogger("aidba.db")


class DatabaseManager:
    """Manages database connections safely."""

    def __init__(self, configs):
        self.connectors = {}
        self.configs = configs if configs else []

        print(f"[AIDBA] DatabaseManager: Loading {len(self.configs)} config(s)", flush=True)

        if not self.configs:
            print("[AIDBA] WARNING: No databases configured", flush=True)
            return

        for idx, c in enumerate(self.configs):
            try:
                # Convert to dict - handles dict, object, pydantic model
                if isinstance(c, dict):
                    cfg = dict(c)
                elif hasattr(c, "model_dump"):
                    cfg = c.model_dump()
                elif hasattr(c, "dict"):
                    cfg = c.dict()
                else:
                    # Try to convert any object with attributes
                    cfg = {
                        "name": getattr(c, "name", f"db_{idx}"),
                        "type": getattr(c, "type", "sqlserver"),
                        "host": getattr(c, "host", "localhost"),
                        "port": getattr(c, "port", 1433),
                        "database": getattr(c, "database", "master"),
                        "username": getattr(c, "username", ""),
                        "password": getattr(c, "password", ""),
                        "enabled": getattr(c, "enabled", True),
                    }
                    # Also try vars() for any object
                    if not cfg.get("name") or cfg["name"] == f"db_{idx}":
                        try:
                            for k, v in vars(c).items():
                                if not k.startswith("_") and k in ["name", "type", "host", "port", "database", "username", "password", "enabled"]:
                                    cfg[k] = v
                        except Exception:
                            pass

                # Validate required fields
                if not isinstance(cfg, dict):
                    print(f"[AIDBA] Skipping config #{idx}: cannot convert {type(c).__name__} to dict", flush=True)
                    continue

                # Get values safely with defaults
                name = str(cfg.get("name", f"db_{idx}"))
                db_type = str(cfg.get("type", "sqlserver")).lower()
                host = str(cfg.get("host", "localhost"))
                port = int(cfg.get("port", 1433))
                database = str(cfg.get("database", "master"))
                username = str(cfg.get("username", ""))
                password = str(cfg.get("password", ""))
                enabled = bool(cfg.get("enabled", True))

                if not enabled:
                    print(f"[AIDBA] Skipping disabled: {name}", flush=True)
                    continue

                print(f"[AIDBA] [{idx+1}/{len(self.configs)}] Connecting to {db_type}: {name} at {host}:{port}", flush=True)

                # Build config dict for connector
                conn_cfg = {
                    "name": name,
                    "type": db_type,
                    "host": host,
                    "port": port,
                    "database": database,
                    "username": username,
                    "password": password,
                    "enabled": enabled,
                }

                if db_type == "sqlserver":
                    from .sqlserver import SqlServerConnector
                    self.connectors[name] = SqlServerConnector(conn_cfg).connect()
                elif db_type == "mysql":
                    from .mysql import MySQLConnector
                    self.connectors[name] = MySQLConnector(conn_cfg).connect()
                elif db_type == "postgresql":
                    from .postgresql import PostgresConnector
                    self.connectors[name] = PostgresConnector(conn_cfg).connect()
                else:
                    print(f"[AIDBA] Unknown DB type: {db_type}", flush=True)
                    continue

                print(f"[AIDBA] ✓ Connected to {db_type}: {name}", flush=True)

            except Exception as e:
                name = "unknown"
                try:
                    if 'cfg' in dir() and isinstance(cfg, dict):
                        name = cfg.get("name", "unknown")
                except Exception:
                    pass
                print(f"[AIDBA] ✗ Failed to connect to {name}: {e}", flush=True)
                log.debug(traceback.format_exc())

        print(f"[AIDBA] Final: {len(self.connectors)} database(s) connected: {list(self.connectors.keys())}", flush=True)

    def list(self):
        return list(self.connectors.keys())

    def get(self, name):
        return self.connectors.get(name)

    def health_all(self):
        result = {}
        for name, conn in self.connectors.items():
            try:
                result[name] = conn.health_check()
            except Exception as e:
                result[name] = {"ok": False, "error": str(e)}
        return result

    def close(self):
        for conn in self.connectors.values():
            try:
                conn.disconnect()
            except Exception:
                pass
