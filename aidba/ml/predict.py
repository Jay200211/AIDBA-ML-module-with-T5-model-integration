"""SQL Predictor - uses the trained T5 model for inference."""
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger("aidba.ml.predict")


class SQLPredictor:
    """Wrapper around the trained T5 model for SQL generation."""

    _instance = None

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the predictor.

        Args:
            model_path: Path to trained model. If None, looks in default locations.
        """
        if model_path is None:
            # Default locations to look for the model
            base_dir = Path(__file__).parent.parent.parent
            candidates = [
                base_dir / "models" / "aidba-sql-t5",
                base_dir / "models" / "t5-small-finetuned",
                base_dir / "aidba-sql-t5",
            ]
            for cand in candidates:
                if cand.exists():
                    model_path = str(cand)
                    break

        self.model_path = model_path
        self.model = None
        self.loaded = False

        if model_path:
            self._load_model()

    def _load_model(self):
        """Load the T5 model."""
        try:
            from aidba.ml.model import T5SQLModel
            self.model = T5SQLModel(model_path=self.model_path)
            self.loaded = self.model.load()
            if self.loaded:
                log.info(f"SQL Predictor loaded from: {self.model_path}")
        except Exception as e:
            log.warning(f"Could not load T5 model: {e}")
            log.warning("Falling back to rule-based approach")
            self.loaded = False

    def predict(self, question: str, schema: str = "") -> Optional[str]:
        """Predict SQL from a natural language question.

        Args:
            question: Natural language question
            schema: Database schema (optional)

        Returns:
            Predicted SQL query or None if failed
        """
        if not self.loaded or self.model is None:
            return None

        return self.model.generate_sql(question, schema)

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> "SQLPredictor":
        """Get or create the global predictor instance."""
        if cls._instance is None:
            cls._instance = SQLPredictor(model_path)
        return cls._instance
