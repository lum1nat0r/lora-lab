import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from lora_lab.scorers.base import ImageRef
from lora_lab.scorers.vlm import VLMJudge, VLMSettings


class Attr:
    def __init__(self, id: str, question: str) -> None:
        self.id = id
        self.question = question
        self.weight = 1.0


ATTRS = [Attr("a", "Is there a vase?"), Attr("b", "Is the background grey?")]


def make_image(tmp_path: Path) -> ImageRef:
    path = tmp_path / "img.png"
    Image.new("RGB", (16, 16), (90, 90, 90)).save(path)
    return ImageRef(
        id=1,
        path=str(path),
        subject="s",
        lora_id="l",
        epoch=1,
        strength=1.0,
        scenario="sc",
        seed=1,
    )


def judge_with(handler, tmp_path: Path) -> tuple[VLMJudge, ImageRef]:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = VLMSettings(base_url="http://judge:8000/v1", model="test-vlm")
    return VLMJudge(settings, client=client), make_image(tmp_path)


def test_parses_json_object(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "test-vlm"
        assert body["messages"][1]["content"][1]["type"] == "image_url"
        content = '```json\n{"a": true, "b": false}\n```'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    judge, image = judge_with(handler, tmp_path)
    verdicts = judge.judge(image, ATTRS)
    assert verdicts == {"a": True, "b": False}
    assert judge.stats == {"requests": 1, "parsed": 1, "refused": 0, "errors": 0}


def test_coerces_strings_and_unknown(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"a": "yes", "b": null}'}}]},
        )

    judge, image = judge_with(handler, tmp_path)
    assert judge.judge(image, ATTRS) == {"a": True, "b": None}


def test_refusal_is_counted_not_raised(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "I'm sorry, I can't help with that."}}]},
        )

    judge, image = judge_with(handler, tmp_path)
    assert judge.judge(image, ATTRS) == {"a": None, "b": None}
    assert judge.stats["refused"] == 1


def test_malformed_answer_counts_as_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "no idea"}}]})

    judge, image = judge_with(handler, tmp_path)
    assert judge.judge(image, ATTRS) == {"a": None, "b": None}
    assert judge.stats["errors"] == 1


def test_http_error_does_not_raise(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    judge, image = judge_with(handler, tmp_path)
    assert judge.judge(image, ATTRS) == {"a": None, "b": None}
    assert judge.stats["errors"] == 1


def test_transient_transport_error_is_retried(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"a": true, "b": false}'}}]}
        )

    judge, image = judge_with(handler, tmp_path)
    assert judge.judge(image, ATTRS) == {"a": True, "b": False}
    assert calls["n"] == 2
    assert judge.stats == {"requests": 1, "parsed": 1, "refused": 0, "errors": 0}


def test_endpoint_normalisation() -> None:
    assert VLMSettings("http://x:8000/v1", "m").endpoint() == "http://x:8000/v1/chat/completions"
    assert (
        VLMSettings("http://x:8000/v1/", "m").endpoint()
        == "http://x:8000/v1/chat/completions"
    )


def test_no_attributes_skips_request(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    judge, image = judge_with(handler, tmp_path)
    assert judge.judge(image, []) == {}
    assert calls == []


@pytest.mark.parametrize("value,expected", [(True, True), ("no", False), ("unknown", None), (1, True)])
def test_coercion_table(value, expected) -> None:
    from lora_lab.scorers.vlm import _coerce

    assert _coerce(value) is expected


def test_vlm_ping_checks_models_endpoint(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"data": [{"id": "test-vlm"}, {"id": "other"}]})

    judge, _ = judge_with(handler, tmp_path)
    status = judge.ping()
    assert status["ok"] is True
    assert status["matched"] is True
    assert "test-vlm" in status["models"]

