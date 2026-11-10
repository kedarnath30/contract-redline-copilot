"""Evaluation harness: score the pipeline against hand-labeled CUAD clauses.

Metrics (all computed per clause, see README for definitions):
  clause_type_accuracy, risk_macro_f1 (+ per-label F1), redline_acceptability,
  over_redline_count, required_redline_recall, citation_correctness, abstention_rate,
  escalation_precision / escalation_recall.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from sklearn.metrics import f1_score

from .assess import FALLBACK, NEEDS_HUMAN, NON_COMPLIANT, RISK_LABELS
from .config import GOLD_PATH, ROOT
from .pipeline import OTHER, Reviewer

NOT_REVIEWED = "not_reviewed"


def load_gold(path: Path | str = GOLD_PATH) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _rubric(item: dict, review, reviewer: Reviewer) -> dict:
    """Automated acceptability rubric for one proposed redline."""
    red = review.redline
    rule = reviewer.playbook.for_type(item["clause_type"])
    cited = [c["rule_id"] for c in red.citations]
    cites_rule = rule is not None and rule.id in cited
    if red.action == "replace":
        uses_language = rule is not None and rule.approved_language in red.proposed
        # Meaning check for a full replacement: it must still be the same kind of clause.
        preserves = reviewer.classifier.predict(red.proposed).label == item["clause_type"]
        warranted = item["risk"] == NON_COMPLIANT
    else:  # suggest: optional insertion that must leave the original wording intact
        uses_language = rule is not None and rule.fallback_language in red.proposed
        preserves = red.proposed.startswith(red.original.rstrip())
        warranted = item["risk"] == FALLBACK
    checks = {
        "cites_rule": cites_rule,
        "uses_approved_language": uses_language,
        "preserves_meaning": preserves,
        "warranted": warranted,
    }
    checks["acceptable"] = all(checks.values())
    return checks


def _citation_ok(item: dict, review, reviewer: Reviewer) -> bool:
    rule = reviewer.playbook.for_type(item["clause_type"])
    cit = review.redline.citations[0]
    span, ev = cit["clause_span"], cit["evidence_span"]
    if rule is None or cit["rule_id"] != rule.id:
        return False
    if (span["start"], span["end"]) != (item["start"], item["end"]):
        return False
    if ev is None:
        return False
    lo, hi = ev["start"] - item["start"], ev["end"] - item["start"]
    return 0 <= lo < hi <= len(item["text"]) and item["text"][lo:hi] == ev["quote"]


def evaluate(gold: list[dict], reviewer: Reviewer | None = None) -> dict:
    reviewer = reviewer or Reviewer()
    t0 = time.perf_counter()
    rows = []
    for item in gold:
        rv = reviewer.review_clause(item["text"], item["start"])
        rows.append((item, rv))
    latency_ms = (time.perf_counter() - t0) * 1000 / max(len(gold), 1)

    n = len(rows)
    type_correct = sum(rv.clause_type == it["clause_type"] for it, rv in rows)
    answered = [(it, rv) for it, rv in rows if not rv.escalated]

    risk_items = [(it, rv) for it, rv in rows if it["risk"] is not None]
    y_true = [it["risk"] for it, _ in risk_items]
    y_pred = [rv.risk or NOT_REVIEWED for _, rv in risk_items]
    labels = list(RISK_LABELS)
    per_label = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    proposals = [
        (it, rv) for it, rv in rows if rv.redline is not None and rv.redline.action != "none"
    ]
    rubrics = [_rubric(it, rv, reviewer) for it, rv in proposals]
    required = [(it, rv) for it, rv in rows if it["risk"] == NON_COMPLIANT]
    over = [
        (it, rv)
        for it, rv in proposals
        if rv.redline.action == "replace" and it["risk"] != NON_COMPLIANT
    ]

    cited = [(it, rv) for it, rv in rows if rv.redline is not None and not rv.escalated]
    citation_ok = [_citation_ok(it, rv, reviewer) for it, rv in cited]

    escalated = [(it, rv) for it, rv in rows if rv.escalated]
    gold_human = [(it, rv) for it, rv in rows if it["risk"] == NEEDS_HUMAN]

    def ratio(a: float, b: float) -> float | None:
        return round(a / b, 4) if b else None

    metrics = {
        "n_clauses": n,
        "clause_type_accuracy": ratio(type_correct, n),
        "clause_type_accuracy_answered": ratio(
            sum(rv.clause_type == it["clause_type"] for it, rv in answered), len(answered)
        ),
        "risk_accuracy": ratio(
            sum(a == b for a, b in zip(y_true, y_pred, strict=True)), len(y_true)
        ),
        "risk_macro_f1": round(float(per_label.mean()), 4),
        **{f"risk_f1_{lab}": round(float(v), 4) for lab, v in zip(labels, per_label, strict=True)},
        "n_redlines_proposed": len(proposals),
        "redline_acceptability": ratio(sum(r["acceptable"] for r in rubrics), len(rubrics)),
        **{
            f"rubric_{k}": ratio(sum(r[k] for r in rubrics), len(rubrics))
            for k in ("cites_rule", "uses_approved_language", "preserves_meaning", "warranted")
        },
        "required_redline_recall": ratio(
            sum(rv.redline is not None and rv.redline.action == "replace" for _, rv in required),
            len(required),
        ),
        "over_redline_count": len(over),
        "citation_correctness": ratio(sum(citation_ok), len(citation_ok)),
        "abstention_rate": ratio(len(escalated), n),
        "escalation_precision": ratio(
            sum(it["risk"] == NEEDS_HUMAN for it, _ in escalated), len(escalated)
        ),
        "escalation_recall": ratio(sum(rv.escalated for _, rv in gold_human), len(gold_human)),
        "mean_latency_ms": round(latency_ms, 2),
    }
    per_clause = [
        {
            "id": it["id"],
            "gold_type": it["clause_type"],
            "pred_type": rv.clause_type,
            "type_confidence": round(rv.type_confidence, 3),
            "gold_risk": it["risk"],
            "pred_risk": rv.risk,
            "escalated": rv.escalated,
            "action": rv.redline.action if rv.redline else None,
            "reason": rv.reason,
        }
        for it, rv in rows
    ]
    return {"metrics": metrics, "per_clause": per_clause}


def log_to_mlflow(result: dict, params: dict, tracking_dir: Path | None = None) -> str | None:
    """Log a run to a local MLflow file store. Returns the run id, or None if not installed."""
    # MLflow 3 puts the plain file store in maintenance mode; it still works and needs no
    # server or database, which keeps CI and local runs dependency-light.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow
    except ImportError:
        return None
    tracking_dir = tracking_dir or ROOT / "mlruns"
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment("redline-eval")
    with mlflow.start_run(run_name=params.get("gold_set", "eval")) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(
            {k: v for k, v in result["metrics"].items() if isinstance(v, int | float)}
        )
        mlflow.log_dict(result, "eval_result.json")
        return run.info.run_id


def gold_type_counts(gold: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for g in gold:
        out[g["clause_type"]] = out.get(g["clause_type"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (kv[0] == OTHER, kv[0])))
