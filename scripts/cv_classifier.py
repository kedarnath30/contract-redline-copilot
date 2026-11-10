"""Contract-grouped 5-fold cross-validation of the clause-type classifier on the weak labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from redline.classifier import ClauseClassifier  # noqa: E402
from redline.config import TRAIN_PATH  # noqa: E402


def main() -> None:
    rows = [json.loads(line) for line in TRAIN_PATH.open()]
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    groups = [r["contract"] for r in rows]
    y_true, y_pred = [], []
    for tr, te in GroupKFold(n_splits=5).split(texts, labels, groups):
        clf = ClauseClassifier().fit([texts[i] for i in tr], [labels[i] for i in tr])
        y_pred += [p.label for p in clf.predict_many([texts[i] for i in te])]
        y_true += [labels[i] for i in te]
    print(
        json.dumps(
            {
                "n": len(y_true),
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            }
        )
    )


if __name__ == "__main__":
    main()
