"""Load the YAML playbook and retrieve the rule that governs a clause."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import PLAYBOOK_PATH

RISK_ORDER = ("unacceptable", "fallback", "preferred")


@dataclass(frozen=True)
class Check:
    id: str
    tier: str  # unacceptable | fallback | preferred
    description: str
    pattern: re.Pattern[str]
    requires_absent: re.Pattern[str] | None = None

    def match(self, text: str) -> re.Match[str] | None:
        if self.requires_absent is not None and self.requires_absent.search(text):
            return None
        return self.pattern.search(text)


@dataclass(frozen=True)
class Rule:
    id: str
    clause_type: str
    title: str
    preferred: str
    fallback: list[str]
    unacceptable: list[str]
    approved_language: str
    fallback_language: str
    checks: list[Check] = field(default_factory=list)

    def as_document(self) -> str:
        """Text used to retrieve this rule (RAG over the playbook)."""
        parts = [self.title, self.preferred, *self.fallback, *self.unacceptable]
        return "\n".join([*parts, self.approved_language])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "clause_type": self.clause_type,
            "title": self.title,
            "preferred": self.preferred,
            "fallback": self.fallback,
            "unacceptable": self.unacceptable,
            "approved_language": self.approved_language,
        }


def _compile(p: str) -> re.Pattern[str]:
    return re.compile(p, re.IGNORECASE | re.DOTALL)


@dataclass
class Playbook:
    version: str
    rules: list[Rule]

    def __post_init__(self) -> None:
        self._by_type = {r.clause_type: r for r in self.rules}
        self._by_id = {r.id: r for r in self.rules}
        self._vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self._matrix = self._vec.fit_transform([r.as_document() for r in self.rules])

    @property
    def clause_types(self) -> list[str]:
        return list(self._by_type)

    def for_type(self, clause_type: str) -> Rule | None:
        return self._by_type.get(clause_type)

    def by_id(self, rule_id: str) -> Rule | None:
        return self._by_id.get(rule_id)

    def retrieve(self, text: str, k: int = 3) -> list[tuple[Rule, float]]:
        """Top-k playbook rules by TF-IDF cosine similarity to the clause text."""
        sims = cosine_similarity(self._vec.transform([text]), self._matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [(self.rules[i], float(sims[i])) for i in order]


def load_playbook(path: Path | str = PLAYBOOK_PATH) -> Playbook:
    raw = yaml.safe_load(Path(path).read_text())
    rules = []
    for r in raw["rules"]:
        checks = []
        for tier in RISK_ORDER:
            for c in r.get("checks", {}).get(tier, []) or []:
                absent = c.get("requires_absent")
                checks.append(
                    Check(
                        id=c["id"],
                        tier=tier,
                        description=c["description"],
                        pattern=_compile(c["pattern"]),
                        requires_absent=_compile(absent) if absent else None,
                    )
                )
        rules.append(
            Rule(
                id=r["id"],
                clause_type=r["clause_type"],
                title=r["title"],
                preferred=r["preferred"].strip(),
                fallback=list(r.get("fallback", [])),
                unacceptable=list(r.get("unacceptable", [])),
                approved_language=" ".join(r["approved_language"].split()),
                fallback_language=" ".join((r.get("fallback_language") or "").split()),
                checks=checks,
            )
        )
    return Playbook(version=str(raw.get("version", "")), rules=rules)


@lru_cache(maxsize=1)
def default_playbook() -> Playbook:
    return load_playbook()
