from starlette.testclient import TestClient
from ia07_embeddings_service import create_app


def test_endpoints():
    c = TestClient(create_app())
    assert c.get("/health").json()["status"] == "UP"
    r = c.post("/v1/embeddings", json={"input": ["hola", "mundo"]}).json()
    assert len(r["data"]) == 2 and len(r["data"][0]["embedding"]) == 512
    rr = c.post("/v1/rerank", json={"query": "pagos stripe",
                "documents": ["stripe adapter cobra", "chat websocket"], "top_n": 2}).json()
    assert rr["results"][0]["index"] == 0
