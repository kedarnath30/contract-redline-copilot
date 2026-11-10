from redline.playbook import RISK_ORDER

EXPECTED = {
    "limitation_of_liability",
    "indemnification",
    "governing_law",
    "auto_renewal",
    "termination_for_convenience",
    "non_compete",
    "ip_assignment",
    "confidentiality",
}


def test_playbook_covers_required_clause_types(playbook):
    assert set(playbook.clause_types) == EXPECTED
    for rule in playbook.rules:
        assert rule.approved_language
        assert rule.checks, rule.id
        tiers = [c.tier for c in rule.checks]
        assert tiers == sorted(tiers, key=RISK_ORDER.index), "checks must be ordered by severity"


def test_rule_ids_unique(playbook):
    ids = [r.id for r in playbook.rules]
    assert len(ids) == len(set(ids))


def test_retrieval_finds_matching_rule(playbook):
    top, score = playbook.retrieve("Each party shall indemnify and hold harmless the other")[0]
    assert top.id == "IND-01" and score > 0
