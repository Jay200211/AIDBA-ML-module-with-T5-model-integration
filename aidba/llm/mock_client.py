"""Mock LLM that uses rule-based logic (no Ollama needed)."""
import json
import logging

log = logging.getLogger("aidba.llm.mock")


class MockLLM:
    """Mock LLM that intelligently picks tools based on keywords."""

    async def generate(self, prompt, temperature=0.1):
        """Return JSON tool call based on keyword detection."""
        prompt_lower = prompt.lower()

        # Detect what the user wants
        if "list_tables" in prompt_lower or "list tables" in prompt_lower or \
           "show tables" in prompt_lower or "what tables" in prompt_lower:
            return json.dumps({"tool": "list_tables", "db": ""})
        elif "list_databases" in prompt_lower or "list databases" in prompt_lower or \
             "what databases" in prompt_lower or "connected" in prompt_lower:
            return json.dumps({"tool": "list_databases"})
        elif "get_health" in prompt_lower or "health" in prompt_lower or \
             "status" in prompt_lower:
            return json.dumps({"tool": "get_health", "db": ""})
        elif "get_performance" in prompt_lower or "performance" in prompt_lower or \
             "metrics" in prompt_lower or "cpu" in prompt_lower:
            return json.dumps({"tool": "get_performance", "db": ""})
        elif "get_slow_queries" in prompt_lower or "slow" in prompt_lower:
            return json.dumps({"tool": "get_slow_queries", "db": ""})
        elif "execute_query" in prompt or "SELECT" in prompt.upper():
            return json.dumps({"tool": "answer", "text": "Query execution requires Ollama. Install Ollama for AI-powered queries."})
        else:
            return json.dumps({"tool": "answer", "text": "I can help you with database operations. Try: 'list tables', 'health', 'performance', or 'slow queries'."})

    async def nlq_to_sql(self, question):
        return ""

    async def optimize_sql(self, dialect, query, plan, schema):
        return {
            "issues": [{"type": "mock", "explanation": "Mock LLM - install Ollama for real AI", "severity": "low"}],
            "rewritten_query": query,
            "indexes_to_create": [],
            "indexes_to_drop": [],
            "estimated_impact_pct": 0,
            "confidence": "low",
            "notes": "Mock provider"
        }

    async def health_qa(self, question, snapshot):
        return f"Mock response to: {question}"
