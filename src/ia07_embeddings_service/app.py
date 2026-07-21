"""IA-07 · Microservicio de embeddings self-hosted, API OpenAI-compatible.

Expone /v1/embeddings, /v1/rerank y /health. Por defecto usa el HashingEmbedder
(offline, sin GPU) para que arranque en cualquier lado; en producción se
reemplaza por un modelo BGE/E5 en la RTX 3070 (FP16) manteniendo el mismo contrato,
de modo que mk5-toolkit y los demás proyectos no cambian de cliente.

Ejecutar:  uvicorn ia_toolkit.embeddings_service.app:app --port 8080
"""
from __future__ import annotations

from typing import List, Union

from pydantic import BaseModel

from ia_core.embeddings import Embedder, HashingEmbedder


class EmbeddingsRequest(BaseModel):
    model: str = "hashing-512"
    input: Union[str, List[str]]


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: int = 5


def create_app(embedder: Embedder | None = None):
    # Import local de FastAPI: el paquete se importa sin exigirlo si no se usa el servicio.
    from fastapi import FastAPI

    emb: Embedder = embedder or HashingEmbedder(dim=512)
    app = FastAPI(title="ia-embeddings", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "UP", "gpu": False, "model": "hashing-512", "dim": emb.dim}

    @app.post("/v1/embeddings")
    def embeddings(req: EmbeddingsRequest):
        texts = [req.input] if isinstance(req.input, str) else list(req.input)
        vecs = emb.embed(texts)
        data = [{"object": "embedding", "index": i, "embedding": vecs[i].tolist()}
                for i in range(len(texts))]
        tokens = sum(len(t.split()) for t in texts)
        return {"object": "list", "data": data, "model": req.model,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}

    @app.post("/v1/rerank")
    def rerank(req: RerankRequest):
        qv = emb.embed([req.query])[0]
        dvs = emb.embed(req.documents)
        scores = (dvs @ qv).tolist()
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[: req.top_n]
        return {"results": [{"index": i, "relevance_score": float(s)} for i, s in ranked]}

    return app


# Instancia por defecto para `uvicorn ...:app`
app = None
try:  # pragma: no cover - solo si FastAPI está instalado
    app = create_app()
except Exception:
    app = None
