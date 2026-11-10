"""End-to-end review: segment -> classify -> retrieve rule -> assess -> redline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .assess import NEEDS_HUMAN, Assessment
from .backends import Backend, OfflineBackend
from .classifier import ClauseClassifier, default_classifier
from .config import Settings, get_settings
from .playbook import Playbook, default_playbook
from .redline import Redline, build_redline
from .segment import segment

OTHER = "other"


@dataclass(frozen=True)
class ClauseReview:
    clause_start: int
    clause_end: int
    clause_type: str
    type_confidence: float
    retrieved_rules: list[dict]
    risk: str | None
    escalated: bool
    reason: str
    redline: Redline | None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class Reviewer:
    def __init__(
        self,
        playbook: Playbook | None = None,
        classifier: ClauseClassifier | None = None,
        backend: Backend | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.playbook = playbook or default_playbook()
        self.classifier = classifier or default_classifier()
        self.backend = backend or OfflineBackend()

    def review_clause(self, text: str, start: int = 0) -> ClauseReview:
        pred = self.classifier.predict(text)
        retrieved = [
            {"rule_id": r.id, "clause_type": r.clause_type, "score": round(s, 4)}
            for r, s in self.playbook.retrieve(text, k=3)
        ]
        end = start + len(text)
        governed = max((v for k, v in pred.scores.items() if k != OTHER), default=0.0)
        uncertain = pred.confidence < self.settings.min_type_confidence
        if uncertain and governed < self.settings.min_governed_score:
            # Uncertain, but no playbook clause type is a real candidate: likely boilerplate.
            return ClauseReview(
                start,
                end,
                OTHER,
                pred.confidence,
                retrieved,
                None,
                False,
                f"Outside the playbook (uncertain type; best governed p={governed:.2f}).",
                None,
            )
        if uncertain:
            return ClauseReview(
                start,
                end,
                pred.label,
                pred.confidence,
                retrieved,
                NEEDS_HUMAN,
                True,
                f"Escalated: clause type uncertain (p={pred.confidence:.2f}).",
                None,
            )
        rule = self.playbook.for_type(pred.label)
        if pred.label == OTHER or rule is None:
            return ClauseReview(
                start,
                end,
                pred.label,
                pred.confidence,
                retrieved,
                None,
                False,
                "Not covered by the playbook.",
                None,
            )
        assessment: Assessment
        assessment, proposed = self.backend.review(text, rule)
        redline = build_redline(text, start, rule, assessment, proposed)
        return ClauseReview(
            start,
            end,
            pred.label,
            pred.confidence,
            retrieved,
            assessment.risk,
            assessment.escalated,
            assessment.reason,
            redline,
        )

    def review_document(self, text: str, include_other: bool = False) -> dict:
        reviews = [self.review_clause(s.text, s.start) for s in segment(text)]
        kept = [r for r in reviews if include_other or r.clause_type != OTHER or r.escalated]
        summary: dict[str, int] = {}
        for r in reviews:
            key = r.risk or "not_reviewed"
            summary[key] = summary.get(key, 0) + 1
        return {
            "playbook_version": self.playbook.version,
            "backend": self.backend.name,
            "n_segments": len(reviews),
            "summary": summary,
            "clauses": [r.to_dict() for r in kept],
        }
