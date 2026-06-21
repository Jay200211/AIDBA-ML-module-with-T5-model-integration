"""Ollama HTTP client for local LLM."""
import json
import logging
import hashlib
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger("aidba.llm.ollama")


class OllamaClient:
    """Talk to local Ollama over HTTP."""

    def __init__(self, base_url="http://localhost:11434", model="deepseek-coder:6.7b", timeout=120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._cache = {}

    async def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            async with httpx.AsyncClient(timeout=5) as cx:
                r = await cx.get(f"{self.base_url}/api/tags")
                if r.status_code == 200:
                    models = [m["name"] for m in r.json().get("models", [])]
                    return any(self.model in m for m in models)
        except Exception as e:
            log.warning(f"Ollama not available: {e}")
        return False

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=3))
    async def generate(self, prompt, temperature=0.1):
        """Generate text from prompt."""
        cache_key = hashlib.sha1((self.model + prompt).encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as cx:
                r = await cx.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": temperature}
                    }
                )
                r.raise_for_status()
                text = (r.json().get("response") or "").strip()
                self._cache[cache_key] = text
                return text
        except Exception as e:
            log.error(f"Ollama generate failed: {e}")
            return ""

    async def nlq_to_sql(self, question):
        """Generate SQL from question - kept for compatibility."""
        return await self.generate(question, temperature=0.0)

    async def optimize_sql(self, dialect, query, plan, schema):
        return {"issues": [], "notes": "Use mock for now"}

    async def health_qa(self, question, snapshot):
        return f"Question: {question}\nSnapshot: {json.dumps(snapshot)[:200]}"
