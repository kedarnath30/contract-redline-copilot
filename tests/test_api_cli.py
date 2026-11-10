import json

from fastapi.testclient import TestClient

from redline.api import app
from redline.cli import main

client = TestClient(app)


def test_health_and_playbook():
    assert client.get("/health").json()["status"] == "ok"
    rules = client.get("/playbook").json()["rules"]
    assert {r["id"] for r in rules} >= {"LOL-01", "GOV-01"}


def test_review_endpoint_returns_cited_redline():
    text = (
        "12. GOVERNING LAW. This Agreement shall be governed by the laws of England and Wales.\n\n"
        "13. NOTICES. All notices under this Agreement shall be in writing."
    )
    body = client.post("/review", json={"text": text}).json()
    gov = [c for c in body["clauses"] if c["clause_type"] == "governing_law"]
    assert gov and gov[0]["risk"] == "non_compliant"
    red = gov[0]["redline"]
    assert red["action"] == "replace" and red["citations"][0]["rule_id"] == "GOV-01"
    span = red["citations"][0]["evidence_span"]
    assert text[span["start"] : span["end"]] == span["quote"]


def test_review_rejects_blank():
    assert client.post("/review", json={"text": "   "}).status_code == 422


def test_cli_eval_writes_report(tmp_path, capsys):
    out = tmp_path / "eval.json"
    assert main(["eval", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["metrics"]["n_clauses"] >= 40
