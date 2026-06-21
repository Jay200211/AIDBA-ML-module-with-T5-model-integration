"""AIDBA launcher - simple, no input prompts."""
import asyncio
import socket
import sys
import uvicorn
from contextlib import asynccontextmanager
from aidba.config import load_config
from aidba.logging_setup import setup_logging
from aidba.api.server import build_app
from aidba.monitor.collector import Monitor


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        try:
            s.close()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app):
    """Minimal lifespan - just log and yield."""
    print("[AIDBA] Server started", flush=True)
    yield
    print("[AIDBA] Server stopped", flush=True)


def main():
    print("[AIDBA] Starting...", flush=True)
    
    try:
        cfg = load_config("config.yaml")
    except Exception as e:
        print(f"[AIDBA] Config error: {e}", flush=True)
        sys.exit(1)
    
    log = setup_logging(cfg.app.log_level, cfg.app.data_dir / "logs")
    log.info("AIDBA starting")
    
    app = build_app(cfg)
    app.router.lifespan_context = lifespan
    
    port = cfg.app.port
    print("")
    print("=" * 60)
    print("  AIDBA - Autonomous Database Administrator")
    print("=" * 60)
    print(f"  Dashboard: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop")
    print("=" * 60)
    print("")
    sys.stdout.flush()
    
    # Run server - NO input prompt
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
