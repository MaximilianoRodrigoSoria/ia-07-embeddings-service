"""Núcleo compartido vendorizado (subset usado por este proyecto)."""
from .embeddings import Embedder, HashingEmbedder, RemoteEmbedder
__all__ = ['Embedder', 'HashingEmbedder', 'RemoteEmbedder']
