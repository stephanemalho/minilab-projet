import asyncio
import json

import torch
from fastapi.testclient import TestClient

from packages.minilab import server
from packages.minilab.model import Config, MiniLabModel
from packages.minilab.tokenizer import Tokenizer, train_bpe

TOKENIZER = Tokenizer(train_bpe("qui est le chat ? le chat dort. " * 30, vocab_size=300))
torch.manual_seed(0)
ENGINE = server.Engine(
    MiniLabModel(Config(vocab_size=300, context_length=96, d_model=32, n_head=2, n_layer=1)).eval(),
    TOKENIZER,
)
server.app.dependency_overrides[server.get_engine] = lambda: ENGINE
client = TestClient(server.app)
BODY = {"messages": [{"role": "user", "content": "qui est le chat ?"}], "max_tokens": 8}


def events(body):
    with client.stream("POST", "/v1/chat", json=body) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        return [line for line in response.iter_lines() if line]


def test_streaming_sends_one_event_per_token_then_done():
    lines = events(BODY)
    assert all(line.startswith("data: ") for line in lines) and lines[-1] == "data: [DONE]"
    parsed = [json.loads(line[6:]) for line in lines[:-1]]
    deltas, done = parsed[:-1], parsed[-1]
    assert {d["type"] for d in deltas} == {"delta"} and done["type"] == "done"
    assert done["finish_reason"] in {"stop", "length"}
    assert done["usage"]["completion_tokens"] <= 8 and done["usage"]["prompt_tokens"] > 0
    assert done["ttft_ms"] >= 0 and done["total_ms"] >= done["ttft_ms"]


def test_without_streaming_the_same_answer_arrives_in_one_object():
    lines = events(BODY)
    streamed = "".join(json.loads(line[6:])["text"] for line in lines[:-2])
    plain = client.post("/v1/chat", json={**BODY, "stream": False}).json()
    assert plain["text"] == streamed  # greedy: the same text both ways
    assert "ttft_ms" not in plain and set(plain["usage"]) == {"prompt_tokens", "completion_tokens"}


def test_generation_stops_when_the_client_disconnects():
    calls = 0

    async def disconnected() -> bool:
        nonlocal calls
        calls += 1
        return calls > 2

    async def collect():
        body = server.ChatBody.model_validate({**BODY, "max_tokens": 20})
        return [event async for event in server.answer_events(ENGINE, body, disconnected)]

    collected = asyncio.run(collect())
    assert len(collected) <= 2 and all(event["type"] == "delta" for event in collected)


def test_errors_are_explained():
    assert client.post("/v1/chat", json={"messages": []}).status_code == 422
    too_long = {"messages": [{"role": "user", "content": "le chat dort. " * 30}], "stream": False}
    assert client.post("/v1/chat", json=too_long).status_code == 400


def test_missing_checkpoint_gives_503(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CHAT_CHECKPOINT", tmp_path / "absent.pt")
    server.get_engine.cache_clear()
    del server.app.dependency_overrides[server.get_engine]
    try:
        response = client.post("/v1/chat", json=BODY)
        assert response.status_code == 503 and "étape 5" in response.json()["detail"]
    finally:
        server.app.dependency_overrides[server.get_engine] = lambda: ENGINE
        server.get_engine.cache_clear()


def test_the_course_pages_may_call_the_local_server():
    for origin in (
        "http://localhost:3000",
        "https://minilab-llm.com",
        "https://minilab-llm.vercel.app",
    ):
        preflight = client.options(
            "/v1/chat",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        assert preflight.headers["access-control-allow-origin"] == origin
        assert preflight.headers["access-control-allow-private-network"] == "true"
    other = client.get("/health", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in other.headers
