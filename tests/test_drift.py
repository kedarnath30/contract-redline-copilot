import random

from redline.cli import NEW_DIR, REF_DIR
from redline.drift import (
    compare,
    corrupt_ocr,
    load_texts,
    make_shifted_batch,
    noise_score,
    profile_batch,
    psi,
)


def test_psi_zero_for_identical_and_positive_for_shift():
    a = {"x": 50, "y": 50}
    assert psi(a, a) == 0
    assert psi(a, {"x": 95, "y": 5}) > 0.2


def test_noise_score_detects_corruption():
    clean = "The Supplier shall deliver the goods on time and in good condition."
    noisy = corrupt_ocr(clean * 3, 0.3, random.Random(0))
    assert noise_score(clean) == 0
    assert noise_score(noisy) > noise_score(clean)


def test_clean_batch_ok_and_shifted_batch_alerts(classifier):
    ref = profile_batch(load_texts(REF_DIR), classifier)
    new_texts = load_texts(NEW_DIR)
    clean = compare(ref, profile_batch(new_texts, classifier))
    shifted = compare(
        ref, profile_batch(make_shifted_batch(new_texts, classifier=classifier), classifier)
    )
    assert clean["status"] == "OK"
    assert shifted["status"] == "ALERT"
    assert "ocr_noise_rate" in shifted["alerts"] and "clause_type_psi" in shifted["alerts"]
