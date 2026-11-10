from pathlib import Path

from redline.config import Settings
from redline.pipeline import Reviewer

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "cuad" / "contracts" / "new"


def test_review_clause_end_to_end(reviewer):
    clause = "This Agreement shall be governed by the laws of England and Wales."
    r = reviewer.review_clause(clause, start=10)
    assert r.clause_type == "governing_law"
    assert r.risk == "non_compliant"
    assert r.redline.action == "replace"
    assert r.retrieved_rules[0]["rule_id"] == "GOV-01"


def test_low_confidence_governed_clause_escalates(playbook, classifier):
    strict = Reviewer(playbook, classifier, settings=Settings(min_type_confidence=1.01))
    r = strict.review_clause("This Agreement is governed by the laws of the State of Delaware.")
    assert r.escalated and r.risk == "needs_human" and r.redline is None


def test_low_confidence_boilerplate_is_not_escalated(playbook, classifier):
    strict = Reviewer(playbook, classifier, settings=Settings(min_type_confidence=1.01))
    r = strict.review_clause("The parties shall cooperate in good faith.")
    assert not r.escalated and r.clause_type == "other" and r.risk is None


def test_review_document_on_real_cuad_contract(reviewer):
    path = sorted(SAMPLE.glob("*.txt"))[0]
    out = reviewer.review_document(path.read_text())
    assert out["n_segments"] > 5
    assert out["backend"] == "offline"
    text = path.read_text()
    for c in out["clauses"]:
        assert 0 <= c["clause_start"] < c["clause_end"] <= len(text)
