"""FastAPI entry point for the Agent orchestration service."""

from typing import Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import AgentSettings, get_settings
from .routers import browser, workflows

# Import agent router if it exists
try:
    from .routers import agent
    HAS_AGENT_ROUTER = True
except ImportError:
    HAS_AGENT_ROUTER = False


def create_app(settings: Optional[AgentSettings] = None) -> FastAPI:
    """Create a FastAPI application configured for MCP-backed agents."""

    resolved_settings = settings or get_settings()

    app = FastAPI(title=resolved_settings.app_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(browser.router)
    app.include_router(workflows.router)
    if HAS_AGENT_ROUTER:
        app.include_router(agent.router)

    @app.get("/health", tags=["health"])
    def health_check() -> Dict[str, Optional[str]]:
        """Report service status and whether an MCP server is configured."""

        return {
            "status": "ok",
            "service": resolved_settings.app_name,
            "mcp": resolved_settings.rube_mcp_server,
        }

    @app.get("/ready", tags=["health"])
    def readiness_check() -> Dict[str, object]:
        """Readiness check endpoint for Kubernetes."""

        return {
            "status": "ready",
            "service": resolved_settings.app_name,
            "orchestration_ready": True,
            "mcp_configured": bool(resolved_settings.rube_mcp_server)
        }

    @app.get("/metrics", tags=["monitoring"])
    def metrics() -> Dict[str, object]:
        """Basic metrics endpoint."""

        return {
            "service": resolved_settings.app_name,
            "version": "0.1.0",
            "capabilities": ["orchestration", "browser_automation", "workflow_engine", "langchain_agents"]
        }

    return app


app = create_app()

