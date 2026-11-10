import json

import httpx
import pytest

from redline.backends import LLMBackend, OfflineBackend, make_backend, validate_llm_answer
from redline.config import Settings

CLAUSE = "This Agreement shall be governed by the laws of England and Wales."


def _client(answer: dict, seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        content = json.dumps(answer)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def _settings(**kw):
    return Settings(backend="llm", llm_api_key="test-key", **kw)


def test_default_backend_is_offline():
    assert isinstance(make_backend(Settings(backend="offline")), OfflineBackend)


def test_llm_backend_requires_key():
    with pytest.raises(ValueError):
        LLMBackend(Settings(backend="llm", llm_api_key=""))


def test_llm_answer_accepted_and_grounded(playbook):
    rule = playbook.for_type("governing_law")
    seen: list = []
    answer = {
        "risk": "non_compliant",
        "reason": "English law is not a U.S. law.",
        "evidence_quote": "laws of England and Wales",
        "proposed_text": rule.approved_language,
    }
    backend = LLMBackend(_settings(), client=_client(answer, seen))
    a, proposed = backend.review(CLAUSE, rule)
    assert a.risk == "non_compliant" and proposed == rule.approved_language
    assert CLAUSE[a.evidence_start : a.evidence_end] == "laws of England and Wales"
    assert seen[0].headers["authorization"] == "Bearer test-key"
    assert seen[0].url.path.endswith("/chat/completions")


def test_azure_request_shape(playbook):
    seen: list = []
    answer = {"risk": "compliant", "reason": "ok", "evidence_quote": "England"}
    s = _settings(llm_provider="azure", llm_base_url="https://x.openai.azure.com", llm_model="dep")
    LLMBackend(s, client=_client(answer, seen)).review(CLAUSE, playbook.for_type("governing_law"))
    assert seen[0].headers["api-key"] == "test-key"
    assert "/openai/deployments/dep/chat/completions" in str(seen[0].url)


def test_ungrounded_quote_escalates(playbook):
    rule = playbook.for_type("governing_law")
    a, _ = validate_llm_answer(CLAUSE, rule, {"risk": "compliant", "evidence_quote": "Delaware"})
    assert a.risk == "needs_human"


def test_invalid_label_escalates(playbook):
    rule = playbook.for_type("governing_law")
    a, _ = validate_llm_answer(CLAUSE, rule, {"risk": "probably fine", "evidence_quote": "laws"})
    assert a.risk == "needs_human"


def test_non_approved_rewrite_falls_back_to_playbook_language(playbook):
    rule = playbook.for_type("governing_law")
    data = {"risk": "non_compliant", "evidence_quote": "England", "proposed_text": "Use Texas law."}
    a, proposed = validate_llm_answer(CLAUSE, rule, data)
    assert a.risk == "non_compliant" and proposed is None  # caller uses approved language


def test_http_error_escalates(playbook):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    a, _ = LLMBackend(_settings(), client=client).review(CLAUSE, playbook.for_type("governing_law"))
    assert a.risk == "needs_human"
