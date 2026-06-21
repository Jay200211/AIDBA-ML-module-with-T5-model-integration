"""Structured JSON logging."""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            payload.update(record.extra_data)
        return json.dumps(payload)


def setup_logging(level="INFO", log_dir=None):
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(JsonFormatter())
    root.addHandler(sh)
    if log_dir:
        fh = logging.FileHandler(Path(log_dir) / "aidba.log", encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)
    return logging.getLogger("aidba")
