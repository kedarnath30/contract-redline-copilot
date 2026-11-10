from redline.segment import segment

DOC = """MASTER AGREEMENT

1. DEFINITIONS

"Services" means the services described in Exhibit A, as amended from time to time.

12. GOVERNING LAW   This Agreement shall be governed by the laws of the State of Delaware.\
   13. NOTICES   All notices shall be in writing and delivered to the addresses above.

  7
"""


def test_offsets_round_trip():
    for s in segment(DOC):
        assert DOC[s.start : s.end] == s.text


def test_headings_merge_and_inline_sections_split():
    segs = segment(DOC)
    texts = [s.text for s in segs]
    assert any("1. DEFINITIONS" in t and "Services" in t for t in texts)
    assert any(t.startswith("12. GOVERNING LAW") and "Delaware" in t for t in texts)
    assert any(t.startswith("13. NOTICES") for t in texts)
    assert not any("Delaware" in t and "NOTICES" in t for t in texts)


def test_drops_page_numbers():
    assert all(s.text.strip() != "7" for s in segment(DOC))
