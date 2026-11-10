import pytest

from redline.config import GOLD_PATH
from redline.evaluate import evaluate, load_gold, log_to_mlflow


def test_gold_set_is_well_formed():
    gold = load_gold(GOLD_PATH)
    assert len(gold) >= 40
    for g in gold:
        assert g["text"] and g["end"] > g["start"]
        assert g["risk"] in {None, "compliant", "fallback", "non_compliant", "needs_human"}
        assert (g["risk"] is None) == (g["clause_type"] == "other")


def test_evaluate_metrics_in_range(reviewer):
    res = evaluate(load_gold(GOLD_PATH)[:20], reviewer)
    m = res["metrics"]
    for k in ("clause_type_accuracy", "risk_macro_f1", "abstention_rate"):
        assert 0.0 <= m[k] <= 1.0
    assert len(res["per_clause"]) == 20


def test_full_gold_quality_floor(reviewer):
    """Regression gate: fails if a change makes the offline pipeline clearly worse."""
    m = evaluate(load_gold(GOLD_PATH), reviewer)["metrics"]
    assert m["clause_type_accuracy"] >= 0.85
    assert m["citation_correctness"] >= 0.9


def test_mlflow_logging(tmp_path, reviewer):
    pytest.importorskip("mlflow")
    res = evaluate(load_gold(GOLD_PATH)[:5], reviewer)
    run_id = log_to_mlflow(res, {"gold_set": "test"}, tracking_dir=tmp_path / "mlruns")
    assert run_id
    assert any((tmp_path / "mlruns").rglob("meta.yaml"))
