"""Deterministic risk assessment of one clause against one playbook rule."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .playbook import Rule

COMPLIANT, FALLBACK, NON_COMPLIANT, NEEDS_HUMAN = (
    "compliant",
    "fallback",
    "non_compliant",
    "needs_human",
)
RISK_LABELS = (COMPLIANT, FALLBACK, NON_COMPLIANT, NEEDS_HUMAN)
_TIER_TO_RISK = {"unacceptable": NON_COMPLIANT, "fallback": FALLBACK, "preferred": COMPLIANT}

# Governance: things a reviewer must see before any position is taken.
_REDACTED = re.compile(r"\[\s*\*[\s*.]*\]|\[\.\.\.\*+\.\.\.\]")
_BY_REFERENCE = re.compile(r"incorporated\s+(in|into)?\s*.{0,40}by reference", re.I | re.S)
_CUT_AT_PAGE_BREAK = re.compile(r"[A-Za-z,]\s+\d{1,3}\s*$")


@dataclass(frozen=True)
class Assessment:
    risk: str
    rule_id: str | None
    check_id: str | None
    reason: str
    evidence_start: int | None = None  # offsets relative to the clause text
    evidence_end: int | None = None

    @property
    def escalated(self) -> bool:
        return self.risk == NEEDS_HUMAN


def governance_flags(text: str) -> list[str]:
    flags = []
    if _REDACTED.search(text):
        flags.append("a material term is redacted in the source")
    if _BY_REFERENCE.search(text):
        flags.append("terms are incorporated from another document")
    if _CUT_AT_PAGE_BREAK.search(text):
        flags.append("clause appears cut off at a page break")
    return flags


def assess(text: str, rule: Rule) -> Assessment:
    flags = governance_flags(text)
    if flags:
        return Assessment(NEEDS_HUMAN, rule.id, None, "Escalated: " + "; ".join(flags) + ".")
    for check in rule.checks:  # ordered unacceptable -> fallback -> preferred
        m = check.match(text)
        if m:
            return Assessment(
                risk=_TIER_TO_RISK[check.tier],
                rule_id=rule.id,
                check_id=check.id,
                reason=check.description,
                evidence_start=m.start(),
                evidence_end=m.end(),
            )
    return Assessment(
        NEEDS_HUMAN, rule.id, None, "Escalated: no playbook signal matched this clause."
    )
