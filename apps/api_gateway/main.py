"""Main FastAPI entry point."""

import logging
import sys

sys.path.insert(0, "/home/phungkien/ehc_ai_mvp/apps/agent_runtime")
sys.path.insert(0, "/home/phungkien/ehc_ai_mvp/apps/api_gateway")
sys.path.insert(0, "/home/phungkien/ehc_ai_mvp")

from app import app
from shared.py.utils.logging import setup_logging

# Setup logging when module loads
setup_logging(log_level="DEBUG", log_file="logs/api.log")
# Write agent pipeline traces to a separate file for easy filtering
setup_logging(log_level="DEBUG", log_file="logs/agent.log")
# Suppress noisy third-party loggers at INFO level
for _noisy in ("httpx", "httpcore", "uvicorn.access", "hpack", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import uvicorn
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": False,
        "log_level": "info"
    }
    logger.info(f"Starting API Gateway on {config['host']}:{config['port']}")
    uvicorn.run(app, **config)
