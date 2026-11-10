"""Rebuild the shipped CUAD subset from the official CUAD v1 release.

Usage:
    curl -L -o CUAD_v1.zip https://zenodo.org/records/4595826/files/CUAD_v1.zip
    unzip CUAD_v1.zip 'CUAD_v1/CUAD_v1.json' 'CUAD_v1/full_contract_txt/*'
    python scripts/build_cuad_subset.py --cuad-dir CUAD_v1

Outputs (all derived from CUAD v1, CC BY 4.0, The Atticus Project):
    data/cuad/train_clauses.jsonl   weakly labeled clause segments from TRAIN contracts
    data/cuad/heldout_candidates.jsonl  candidate segments from HELD-OUT contracts (gold pool)
    data/cuad/contracts/*.txt       a few full contract texts (reference + new batch for drift)

Labels: a segment gets a clause type when it overlaps a CUAD expert span of a mapped category.
CUAD has no indemnification or confidentiality category, so those two types are labeled by a
heading rule (segment starts with an "Indemnif..." / "Confidential..." heading). That rule is a
weak label and is documented as such in data/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redline.segment import segment  # noqa: E402

CUAD_TO_TYPE = {
    "Cap On Liability": "limitation_of_liability",
    "Uncapped Liability": "limitation_of_liability",
    "Governing Law": "governing_law",
    "Renewal Term": "auto_renewal",
    "Notice Period To Terminate Renewal": "auto_renewal",
    "Termination For Convenience": "termination_for_convenience",
    "Non-Compete": "non_compete",
    "Ip Ownership Assignment": "ip_assignment",
}
OTHER_CATEGORIES = {
    "Audit Rights",
    "Insurance",
    "Anti-Assignment",
    "Exclusivity",
    "License Grant",
    "Minimum Commitment",
    "Revenue/Profit Sharing",
    "Post-Termination Services",
    "Warranty Duration",
    "Liquidated Damages",
    "Change Of Control",
}
HEADING_RULES = {
    "indemnification": re.compile(r"^[\s\d.()A-Za-z]{0,12}indemnif", re.I),
    "confidentiality": re.compile(r"^[\s\d.()A-Za-z]{0,12}confidential", re.I),
}
# EDGAR exhibits carry "CONFIDENTIAL TREATMENT REQUESTED" / redaction banners; not clauses.
_BANNER = re.compile(
    r"treatment|redact|omitted|filed separately|commission|execution copy|confidential project",
    re.I,
)
MAX_CHARS = 2000
PER_CLASS_CAP = 220


def is_heldout(title: str) -> bool:
    """Deterministic ~10% contract-level hold-out, so gold clauses never leak into training."""
    return int(hashlib.md5(title.encode()).hexdigest(), 16) % 10 == 0


def label_segments(doc: dict) -> list[dict]:
    para = doc["paragraphs"][0]
    text = para["context"]
    spans: list[tuple[int, int, str]] = []
    for qa in para["qas"]:
        cat = qa["id"].split("__")[-1]
        for a in qa["answers"]:
            spans.append((a["answer_start"], a["answer_start"] + len(a["text"]), cat))
    out = []
    for seg in segment(text):
        if len(seg.text) > MAX_CHARS:
            continue
        overlap: Counter[str] = Counter()
        for s, e, cat in spans:
            ov = min(seg.end, e) - max(seg.start, s)
            if ov > 0:
                overlap[cat] += ov
        label, source = None, None
        mapped = Counter()
        for cat, ov in overlap.items():
            if cat in CUAD_TO_TYPE:
                mapped[CUAD_TO_TYPE[cat]] += ov
        if mapped:
            label, source = mapped.most_common(1)[0][0], "cuad"
        else:
            for t, rx in HEADING_RULES.items():
                if rx.search(seg.text[:60]) and not _BANNER.search(seg.text[:250]):
                    label, source = t, "heading_rule"
                    break
            if label is None and overlap and set(overlap) & OTHER_CATEGORIES:
                low = seg.text.lower()
                if "indemnif" not in low and "confidential" not in low:
                    label, source = "other", "cuad"
        if label:
            out.append(
                {
                    "contract": doc["title"],
                    "start": seg.start,
                    "end": seg.end,
                    "label": label,
                    "label_source": source,
                    "cuad_categories": sorted(overlap),
                    "text": seg.text,
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuad-dir", type=Path, required=True)
    ap.add_argument("--n-reference", type=int, default=16)
    ap.add_argument("--n-new", type=int, default=8)
    args = ap.parse_args()
    data = json.loads((args.cuad_dir / "CUAD_v1.json").read_text())["data"]

    rng = random.Random(7)
    train, heldout = defaultdict(list), []
    for doc in data:
        rows = label_segments(doc)
        if is_heldout(doc["title"]):
            heldout.extend(rows)
        else:
            for r in rows:
                train[r["label"]].append(r)

    out_dir = ROOT / "data" / "cuad"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "train_clauses.jsonl").open("w") as f:
        for label in sorted(train):
            rows = train[label]
            rng.shuffle(rows)
            for r in rows[:PER_CLASS_CAP]:
                f.write(json.dumps(r) + "\n")
    with (out_dir / "heldout_candidates.jsonl").open("w") as f:
        for r in heldout:
            f.write(json.dumps(r) + "\n")

    # Small full-text samples for the drift demo: reference from train, new batch from held-out.
    contracts = out_dir / "contracts"
    contracts.mkdir(exist_ok=True)
    small = [d for d in data if 8_000 < len(d["paragraphs"][0]["context"]) < 45_000]
    ref = sorted((d for d in small if not is_heldout(d["title"])), key=lambda d: d["title"])
    new = sorted((d for d in small if is_heldout(d["title"])), key=lambda d: d["title"])
    rng.shuffle(ref)
    rng.shuffle(new)
    for split, docs in (("reference", ref[: args.n_reference]), ("new", new[: args.n_new])):
        (contracts / split).mkdir(exist_ok=True)
        for d in docs:
            name = re.sub(r"[^A-Za-z0-9._-]+", "_", d["title"])[:90] + ".txt"
            (contracts / split / name).write_text(d["paragraphs"][0]["context"])

    print({k: min(len(v), PER_CLASS_CAP) for k, v in sorted(train.items())})
    print("heldout", Counter(r["label"] for r in heldout))


if __name__ == "__main__":
    main()
