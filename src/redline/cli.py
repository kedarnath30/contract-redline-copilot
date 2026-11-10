"""Command-line interface: review, eval, drift, report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DATA_DIR, GOLD_PATH, ROOT, get_settings

BLIND_PATH = DATA_DIR / "gold" / "blind.jsonl"
REF_DIR = DATA_DIR / "cuad" / "contracts" / "reference"
NEW_DIR = DATA_DIR / "cuad" / "contracts" / "new"


def _reviewer():
    from .backends import make_backend
    from .pipeline import Reviewer

    settings = get_settings()
    return Reviewer(backend=make_backend(settings), settings=settings)


def cmd_review(args: argparse.Namespace) -> int:
    text = Path(args.file).read_text(errors="replace") if args.file != "-" else sys.stdin.read()
    result = _reviewer().review_document(text, include_other=args.include_other)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"{result['n_segments']} segments | {result['summary']}")
    for c in result["clauses"]:
        rl = c["redline"]
        print(
            f"\n[{c['clause_start']}:{c['clause_end']}] {c['clause_type']} "
            f"(p={c['type_confidence']:.2f}) -> {c['risk']}"
        )
        print(f"  {c['reason']}")
        if rl and rl["action"] != "none":
            print(f"  {rl['action'].upper()}: {rl['rationale']}")
            print(f"  diff: {rl['diff'][:400]}")
    return 0


def run_eval(gold_path: Path, log_mlflow: bool, name: str) -> dict:
    from .evaluate import evaluate, load_gold, log_to_mlflow

    reviewer = _reviewer()
    gold = load_gold(gold_path)
    result = evaluate(gold, reviewer)
    if log_mlflow:
        run_id = log_to_mlflow(
            result,
            {
                "gold_set": name,
                "backend": reviewer.backend.name,
                "playbook_version": reviewer.playbook.version,
                "min_type_confidence": reviewer.settings.min_type_confidence,
                "n_gold": len(gold),
            },
        )
        result["mlflow_run_id"] = run_id
    return result


def cmd_eval(args: argparse.Namespace) -> int:
    result = run_eval(Path(args.gold), args.mlflow, args.name)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["metrics"], indent=2))
    m = result["metrics"]
    if args.min_type_accuracy and m["clause_type_accuracy"] < args.min_type_accuracy:
        print(f"FAIL: clause_type_accuracy < {args.min_type_accuracy}", file=sys.stderr)
        return 1
    if args.min_citation and (m["citation_correctness"] or 0) < args.min_citation:
        print(f"FAIL: citation_correctness < {args.min_citation}", file=sys.stderr)
        return 1
    return 0


def run_drift(reference: Path, new: Path, inject_shift: bool) -> dict:
    from .drift import compare, load_texts, make_shifted_batch, profile_batch

    ref = profile_batch(load_texts(reference))
    texts = load_texts(new)
    if inject_shift:
        texts = make_shifted_batch(texts)
    return compare(ref, profile_batch(texts))


def cmd_drift(args: argparse.Namespace) -> int:
    from .drift import to_markdown

    report = run_drift(Path(args.reference), Path(args.new), args.inject_shift)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    print(to_markdown(report))
    return 2 if (args.fail_on_alert and report["status"] == "ALERT") else 0


def cmd_report(args: argparse.Namespace) -> int:
    from .playbook import default_playbook
    from .report import write_report

    evals = {
        "dev": run_eval(GOLD_PATH, args.mlflow, "dev"),
        "blind": run_eval(BLIND_PATH, args.mlflow, "blind"),
    }
    drift = {
        "clean new batch vs reference": run_drift(REF_DIR, NEW_DIR, False),
        "injected shifted batch vs reference": run_drift(REF_DIR, NEW_DIR, True),
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    for k, v in evals.items():
        (reports / f"eval_{k}.json").write_text(json.dumps(v, indent=2))
    (reports / "drift_clean.json").write_text(
        json.dumps(drift["clean new batch vs reference"], indent=2)
    )
    (reports / "drift_shifted.json").write_text(
        json.dumps(drift["injected shifted batch vs reference"], indent=2)
    )
    meta = {"backend": get_settings().backend, "playbook_version": default_playbook().version}
    out = write_report(Path(args.out), evals, drift, meta)
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="redline", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("review", help="review a contract file (or - for stdin)")
    r.add_argument("file")
    r.add_argument("--json", action="store_true")
    r.add_argument("--include-other", action="store_true")
    r.set_defaults(func=cmd_review)

    e = sub.add_parser("eval", help="score against a labeled gold set")
    e.add_argument("--gold", default=str(GOLD_PATH))
    e.add_argument("--name", default="dev")
    e.add_argument("--out")
    e.add_argument("--mlflow", action="store_true", help="log the run to ./mlruns")
    e.add_argument("--min-type-accuracy", type=float, default=0.0)
    e.add_argument("--min-citation", type=float, default=0.0)
    e.set_defaults(func=cmd_eval)

    d = sub.add_parser("drift", help="compare a new batch of contracts to a reference batch")
    d.add_argument("--reference", default=str(REF_DIR))
    d.add_argument("--new", default=str(NEW_DIR))
    d.add_argument("--inject-shift", action="store_true", help="demo: corrupt and skew the batch")
    d.add_argument("--out")
    d.add_argument("--fail-on-alert", action="store_true")
    d.set_defaults(func=cmd_drift)

    rp = sub.add_parser("report", help="run eval + drift and write docs/index.html")
    rp.add_argument("--out", default=str(ROOT / "docs" / "index.html"))
    rp.add_argument("--mlflow", action="store_true")
    rp.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
