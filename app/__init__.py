"""Agent service package exposing the FastAPI application factory."""

from .main import create_app

__all__ = ["create_app"]

