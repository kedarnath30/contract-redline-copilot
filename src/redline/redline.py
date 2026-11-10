"""Turn an assessment into a proposed edit: a word-level diff plus a cited rationale."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from .assess import FALLBACK, NON_COMPLIANT, Assessment
from .playbook import Rule

_TOKEN = re.compile(r"\S+|\s+")


@dataclass(frozen=True)
class Redline:
    action: str  # none | suggest (optional, fallback) | replace (required, non-compliant)
    original: str
    proposed: str
    diff: str
    rationale: str
    citations: list[dict] = field(default_factory=list)


def _mark(open_: str, chunk: str, close: str, lead: bool = True, trail: bool = False) -> str:
    pre = " " if lead and chunk[:1].isspace() else ""
    post = " " if trail and chunk[-1:].isspace() else ""
    return f"{pre}{open_}{chunk.strip()}{close}{post}"


def word_diff(old: str, new: str) -> str:
    """Inline diff: [-deleted-] {+inserted+}."""
    a, b = _TOKEN.findall(old), _TOKEN.findall(new)
    out: list[str] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append("".join(a[i1:i2]))
            continue
        if op in ("delete", "replace"):
            out.append(_mark("[-", "".join(a[i1:i2]), "-]"))
        if op in ("insert", "replace"):
            out.append(_mark("{+", "".join(b[j1:j2]), "+}", lead=op == "insert", trail=True))
    return "".join(out)


def build_redline(
    clause: str,
    clause_start: int,
    rule: Rule,
    assessment: Assessment,
    proposed_override: str | None = None,
) -> Redline:
    span = None
    if assessment.evidence_start is not None and assessment.evidence_end is not None:
        span = {
            "start": clause_start + assessment.evidence_start,
            "end": clause_start + assessment.evidence_end,
            "quote": clause[assessment.evidence_start : assessment.evidence_end],
        }
    citations = [
        {
            "rule_id": rule.id,
            "check_id": assessment.check_id,
            "clause_span": {"start": clause_start, "end": clause_start + len(clause)},
            "evidence_span": span,
        }
    ]
    if assessment.risk == NON_COMPLIANT:
        proposed = proposed_override or rule.approved_language
        action = "replace"
        why = (
            f"{rule.id} ({rule.title}): {assessment.reason} The playbook lists this as "
            f"unacceptable, so the clause is replaced with the approved language."
        )
    elif assessment.risk == FALLBACK and rule.fallback_language:
        proposed = (proposed_override or f"{clause.rstrip()} {rule.fallback_language}").strip()
        action = "suggest"
        why = (
            f"{rule.id} ({rule.title}): {assessment.reason} This is an acceptable fallback; "
            f"the optional insertion moves it toward the preferred position."
        )
    else:
        return Redline("none", clause, clause, "", f"{rule.id}: {assessment.reason}", citations)
    return Redline(action, clause, proposed, word_diff(clause, proposed), why, citations)
