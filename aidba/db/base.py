"""Abstract base for all DB connectors - with safe cfg handling."""
import abc
from typing import Any, Dict


class BaseConnector(abc.ABC):
    """Base class for database connectors."""

    dialect = "base"

    def __init__(self, cfg: Any = None):
        """Initialize the connector.

        Args:
            cfg: Can be a dict, Pydantic model, or None
        """
        if cfg is None:
            self.cfg = {}
        elif isinstance(cfg, dict):
            self.cfg = cfg
        elif hasattr(cfg, "model_dump"):
            # Pydantic v2
            self.cfg = cfg.model_dump()
        elif hasattr(cfg, "dict"):
            # Pydantic v1
            self.cfg = cfg.dict()
        else:
            # Last resort - try to convert
            try:
                self.cfg = dict(cfg)
            except Exception:
                self.cfg = {}

        self.name = self.cfg.get("name", "unknown") if isinstance(self.cfg, dict) else "unknown"

    def get_cfg(self, key: str, default=None):
        """Safely get a config value."""
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        return default

    @abc.abstractmethod
    def connect(self):
        """Establish the database connection."""
        pass

    @abc.abstractmethod
    def disconnect(self):
        """Close the database connection."""
        pass

    @abc.abstractmethod
    def execute(self, sql, params=None):
        """Execute a SQL query and return results as list of dicts."""
        pass

    @abc.abstractmethod
    def health_check(self):
        """Check if the database is reachable. Returns dict with 'ok' key."""
        pass
