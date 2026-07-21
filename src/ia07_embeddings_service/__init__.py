"""IA-07 · Servicio de embeddings self-hosted (FastAPI, OpenAI-compatible)."""
from .app import create_app

__all__ = ["create_app"]
