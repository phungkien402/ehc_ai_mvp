"""API Gateway application."""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, "/home/phungkien/ehc_ai_mvp")

from app.core.config import config
from shared.py.utils.logging import setup_logging
from app.api.routes import router

# Setup logging
setup_logging(log_level="INFO", log_file="logs/api.log")
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="EHC AI Helpdesk",
    description="Vietnamese helpdesk FAQ chatbot",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)

dashboard_dist = Path("apps/dashboard-ui/dist")
if dashboard_dist.exists():
    app.mount("/ui", StaticFiles(directory=str(dashboard_dist), html=True), name="dashboard-ui")


@app.get("/")
async def root():
    return {
        "service": "EHC AI Helpdesk",
        "status": "running",
        "docs": "/docs",
        "dashboard": "/ui"
    }
