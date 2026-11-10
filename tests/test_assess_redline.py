import pytest

from redline.assess import assess
from redline.redline import build_redline, word_diff


@pytest.mark.parametrize(
    "text,risk,check",
    [
        ("This Agreement is governed by the laws of the State of Delaware.", "compliant", "P1"),
        ("This Agreement is governed by the laws of the State of Texas.", "fallback", "F1"),
        (
            "This Agreement shall be governed by the laws of England and Wales.",
            "non_compliant",
            "U1",
        ),
    ],
)
def test_governing_law_positions(playbook, text, risk, check):
    a = assess(text, playbook.for_type("governing_law"))
    assert a.risk == risk
    assert a.check_id.endswith(check)
    assert text[a.evidence_start : a.evidence_end]


def test_one_sided_limitation_is_non_compliant(playbook):
    text = "In no event shall Supplier be liable for any consequential damages."
    assert assess(text, playbook.for_type("limitation_of_liability")).risk == "non_compliant"


def test_redacted_term_escalates(playbook):
    text = "Either party may terminate upon [***] days notice to the other party."
    a = assess(text, playbook.for_type("termination_for_convenience"))
    assert a.risk == "needs_human" and a.escalated


def test_no_signal_escalates_instead_of_guessing(playbook):
    a = assess("The parties met on Tuesday.", playbook.for_type("ip_assignment"))
    assert a.risk == "needs_human"


def test_replace_redline_uses_approved_language_and_cites(playbook):
    rule = playbook.for_type("governing_law")
    clause = "This Agreement shall be governed by the laws of Spain."
    a = assess(clause, rule)
    red = build_redline(clause, 100, rule, a)
    assert red.action == "replace"
    assert red.proposed == rule.approved_language
    cit = red.citations[0]
    assert cit["rule_id"] == "GOV-01"
    assert cit["clause_span"] == {"start": 100, "end": 100 + len(clause)}
    ev = cit["evidence_span"]
    assert clause[ev["start"] - 100 : ev["end"] - 100] == ev["quote"]
    assert "[-" in red.diff and "{+" in red.diff


def test_fallback_suggestion_keeps_original(playbook):
    rule = playbook.for_type("auto_renewal")
    clause = (
        "This Agreement renews automatically for one year terms unless either party gives "
        "notice of non-renewal sixty (60) days before the end of the term."
    )
    a = assess(clause, rule)
    red = build_redline(clause, 0, rule, a)
    assert a.risk == "fallback" and red.action == "suggest"
    assert red.proposed.startswith(clause) and rule.fallback_language in red.proposed


def test_word_diff():
    assert word_diff("laws of Spain.", "laws of Delaware.") == "laws of [-Spain.-]{+Delaware.+}"
