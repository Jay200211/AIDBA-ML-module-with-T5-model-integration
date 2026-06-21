"""LLM provider factory - manages Ollama connection."""
import logging

log = logging.getLogger("aidba.llm")

_singleton = None


def get_llm(cfg):
    """Get or create the LLM client singleton."""
    global _singleton
    if _singleton is not None:
        return _singleton

    provider = getattr(cfg.llm, "provider", "ollama")

    if provider == "ollama":
        from .ollama_client import OllamaClient
        _singleton = OllamaClient(
            base_url=cfg.llm.base_url,
            model=cfg.llm.model,
            timeout=cfg.llm.timeout_seconds,
        )
        log.info(f"LLM initialized: Ollama ({cfg.llm.model})")
    elif provider == "mock":
        from .mock_client import MockLLM
        _singleton = MockLLM()
        log.info("LLM initialized: Mock provider")
    else:
        log.warning(f"Unknown LLM provider '{provider}', using mock")
        from .mock_client import MockLLM
        _singleton = MockLLM()

    return _singleton


def reset_llm():
    """Reset the LLM singleton."""
    global _singleton
    _singleton = None
