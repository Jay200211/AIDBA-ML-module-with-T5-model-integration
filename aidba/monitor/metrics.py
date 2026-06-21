"""Metric data class."""
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


@dataclass
class MetricPoint:
    db_name: str
    metric: str
    value: float
    labels: dict = field(default_factory=dict)
    ts: str = ""

    def __post_init__(self):
        if not self.ts:
            self.ts = datetime.utcnow().isoformat() + "Z"

    def to_json(self):
        return json.dumps(asdict(self))
