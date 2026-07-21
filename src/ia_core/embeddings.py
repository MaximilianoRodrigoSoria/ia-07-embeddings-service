"""Puerto de embeddings + adaptadores.

`Embedder` es el puerto. Dos adaptadores:

- `HashingEmbedder`: offline, determinista, sin dependencias de modelo. Proyecta
  n-gramas de caracteres a un espacio de dimensión fija con el "hashing trick" y
  normaliza L2. No es semántico de verdad, pero es reproducible y suficiente para
  tests y demos sin GPU. Es el fallback que ya existe conceptualmente en mk5-toolkit.
- `RemoteEmbedder`: cliente OpenAI-compatible (usa el servicio IA-07 local o una
  API). Sin dependencias extra: usa urllib.

En producción se puede agregar un `SentenceTransformerEmbedder` si la librería
está instalada; se deja como ejercicio para no imponer la dependencia pesada.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from typing import Protocol, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    """Puerto: convierte textos en una matriz (n, dim) de vectores L2-normalizados."""

    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Embedder offline y determinista (hashing trick sobre tokens + bigramas)."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        toks = _tokenize(text)
        # Unigramas y bigramas para captar algo de contexto local.
        grams = list(toks)
        grams += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack([self._vector(t) for t in texts])


class RemoteEmbedder:
    """Adaptador OpenAI-compatible (POST /v1/embeddings). Consume IA-07 o una API."""

    def __init__(self, base_url: str, model: str = "bge-m3", dim: int = 1024,
                 api_key: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/v1/embeddings", data=payload, headers=headers
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # pragma: no cover - I/O
            data = json.loads(resp.read())
        rows = [item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"])]
        arr = np.asarray(rows, dtype=np.float64)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
