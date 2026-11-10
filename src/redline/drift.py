"""Data-quality and drift monitor for incoming contract batches.

A batch profile records, per segment: the predicted clause type, classifier confidence, and an
encoding/OCR-noise score. Comparing a new batch with a reference batch yields:

* clause-type distribution shift: Population Stability Index (PSI) and a chi-square test
* unknown-clause rate: share of segments the classifier is not confident about
* OCR / encoding noise rate: share of segments with mojibake, control characters, U+FFFD, or
  OCR digit-for-letter confusions
* mean classifier confidence

Thresholds are conventional starting points (PSI > 0.2 is the usual "significant shift" line)
and live in ``Thresholds`` so a team can tune them per customer.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from scipy.stats import chi2_contingency

from .classifier import ClauseClassifier, default_classifier
from .segment import segment

_MOJIBAKE = re.compile(r"Ã.|â€|Â[^\w\s]?|\ufffd|[\x00-\x08\x0b\x0c\x0e-\x1f]")
_OCR_CONFUSION = re.compile(r"\b[A-Za-z]+[0-9][A-Za-z]+\b|\b[a-z]+[|][a-z]+\b")
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Thresholds:
    psi: float = 0.2
    chi2_p: float = 0.01
    unknown_rate_increase: float = 0.10
    noise_rate_increase: float = 0.05
    confidence_drop: float = 0.10
    min_type_confidence: float = 0.45


def noise_score(text: str) -> float:
    """Noisy characters/tokens per 100 words."""
    words = max(len(text.split()), 1)
    hits = len(_MOJIBAKE.findall(text)) + len(_OCR_CONFUSION.findall(text))
    return 100.0 * hits / words


@dataclass
class BatchProfile:
    n_documents: int
    n_segments: int
    type_counts: dict[str, int]
    unknown_rate: float
    noise_rate: float
    mean_confidence: float
    noisy_examples: list[str] = field(default_factory=list)


def profile_batch(
    texts: list[str],
    classifier: ClauseClassifier | None = None,
    thresholds: Thresholds | None = None,
    noisy_segment_score: float = 2.0,
) -> BatchProfile:
    clf = classifier or default_classifier()
    th = thresholds or Thresholds()
    segs = [s.text for t in texts for s in segment(t)]
    preds = clf.predict_many(segs) if segs else []
    counts: Counter[str] = Counter()
    noisy, conf = [], 0.0
    for seg, p in zip(segs, preds, strict=True):
        counts[p.label if p.confidence >= th.min_type_confidence else UNKNOWN] += 1
        conf += p.confidence
        if noise_score(seg) >= noisy_segment_score:
            noisy.append(seg[:160])
    n = max(len(segs), 1)
    return BatchProfile(
        n_documents=len(texts),
        n_segments=len(segs),
        type_counts=dict(sorted(counts.items())),
        unknown_rate=round(counts[UNKNOWN] / n, 4),
        noise_rate=round(len(noisy) / n, 4),
        mean_confidence=round(conf / n, 4),
        noisy_examples=noisy[:3],
    )


def psi(ref: dict[str, int], new: dict[str, int], eps: float = 1e-4) -> float:
    cats = sorted(set(ref) | set(new))
    rt, nt = sum(ref.values()) or 1, sum(new.values()) or 1
    total = 0.0
    for c in cats:
        r = max(ref.get(c, 0) / rt, eps)
        q = max(new.get(c, 0) / nt, eps)
        total += (q - r) * math.log(q / r)
    return round(total, 4)


def chi_square(ref: dict[str, int], new: dict[str, int]) -> tuple[float, float]:
    cats = sorted(set(ref) | set(new))
    table = [[ref.get(c, 0) + 0.5 for c in cats], [new.get(c, 0) + 0.5 for c in cats]]
    stat, p, _, _ = chi2_contingency(table)
    return round(float(stat), 3), float(p)


def compare(
    reference: BatchProfile, new: BatchProfile, thresholds: Thresholds | None = None
) -> dict:
    th = thresholds or Thresholds()
    psi_val = psi(reference.type_counts, new.type_counts)
    chi_stat, chi_p = chi_square(reference.type_counts, new.type_counts)
    checks = {
        "clause_type_psi": {
            "value": psi_val,
            "threshold": th.psi,
            "alert": psi_val > th.psi,
        },
        "clause_type_chi2": {
            "statistic": chi_stat,
            "p_value": float(f"{chi_p:.3g}"),
            "threshold": th.chi2_p,
            "alert": chi_p < th.chi2_p,
        },
        "unknown_clause_rate": {
            "reference": reference.unknown_rate,
            "new": new.unknown_rate,
            "max_increase": th.unknown_rate_increase,
            "alert": new.unknown_rate - reference.unknown_rate > th.unknown_rate_increase,
        },
        "ocr_noise_rate": {
            "reference": reference.noise_rate,
            "new": new.noise_rate,
            "max_increase": th.noise_rate_increase,
            "alert": new.noise_rate - reference.noise_rate > th.noise_rate_increase,
        },
        "mean_type_confidence": {
            "reference": reference.mean_confidence,
            "new": new.mean_confidence,
            "max_drop": th.confidence_drop,
            "alert": reference.mean_confidence - new.mean_confidence > th.confidence_drop,
        },
    }
    alerts = [k for k, v in checks.items() if v["alert"]]
    return {
        "status": "ALERT" if alerts else "OK",
        "alerts": alerts,
        "checks": checks,
        "reference": asdict(reference),
        "new": asdict(new),
        "thresholds": asdict(th),
    }


def load_texts(directory: Path | str) -> list[str]:
    return [p.read_text(errors="replace") for p in sorted(Path(directory).glob("*.txt"))]


# --- Shift injection for the demo -------------------------------------------------------

_OCR_SWAPS = {"l": "1", "O": "0", "o": "0", "S": "5", "e": "c", "i": "|"}
_MOJI = ["â€™", "Ã©", "\ufffd", "Â§"]


def corrupt_ocr(text: str, rate: float, rng: random.Random) -> str:
    """Simulate a bad scan: OCR letter/digit confusions and encoding mojibake."""
    out = []
    for ch in text:
        r = rng.random()
        if ch in _OCR_SWAPS and r < rate:
            out.append(_OCR_SWAPS[ch])
        elif ch == "'" and r < 0.5:
            out.append(_MOJI[0])
        elif ch == " " and r < rate / 4:
            out.append(rng.choice(_MOJI[1:]) + " ")
        else:
            out.append(ch)
    return "".join(out)


def make_shifted_batch(
    texts: list[str],
    seed: int = 13,
    corrupt_fraction: float = 0.6,
    rate: float = 0.08,
    keep_types: tuple[str, ...] = ("governing_law", "limitation_of_liability", "indemnification"),
    classifier: ClauseClassifier | None = None,
) -> list[str]:
    """Build a shifted batch: OCR-corrupt most documents and skew the clause mix.

    The skew keeps only segments whose predicted type is in ``keep_types`` (plus a few others),
    mimicking a customer who suddenly sends liability-heavy contract excerpts.
    """
    rng = random.Random(seed)
    clf = classifier or default_classifier()
    out = []
    for t in texts:
        segs = [s.text for s in segment(t)]
        preds = clf.predict_many(segs) if segs else []
        kept = [s for s, p in zip(segs, preds, strict=True) if p.label in keep_types]
        kept += [s for s in segs if rng.random() < 0.1]
        doc = "\n\n".join(kept)
        out.append(corrupt_ocr(doc, rate, rng) if rng.random() < corrupt_fraction else doc)
    return out


def to_markdown(report: dict) -> str:
    lines = [f"# Drift report: **{report['status']}**", ""]
    lines.append("| check | detail | alert |")
    lines.append("|---|---|---|")
    for name, c in report["checks"].items():
        detail = ", ".join(f"{k}={v}" for k, v in c.items() if k != "alert")
        lines.append(f"| {name} | {detail} | {'YES' if c['alert'] else 'no'} |")
    lines += ["", "Reference clause mix: " + str(report["reference"]["type_counts"])]
    lines.append("New clause mix: " + str(report["new"]["type_counts"]))
    return "\n".join(lines) + "\n"
