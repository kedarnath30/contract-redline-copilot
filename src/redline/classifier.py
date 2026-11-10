"""Clause-type classifier: TF-IDF + logistic regression, trained on weakly labeled CUAD clauses.

This is the offline baseline. It returns calibrated-ish probabilities (LogReg softmax); the
pipeline treats a low top probability as "unknown clause" and escalates instead of guessing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import TRAIN_PATH


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]


def load_training(path: Path = TRAIN_PATH) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


class ClauseClassifier:
    def __init__(self, seed: int = 0) -> None:
        features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True
                    ),
                ),
            ]
        )
        self.model = Pipeline(
            [
                ("tfidf", features),
                (
                    "clf",
                    LogisticRegression(
                        C=8.0, max_iter=2000, class_weight="balanced", random_state=seed
                    ),
                ),
            ]
        )
        self.labels: list[str] = []

    def fit(self, texts: list[str], labels: list[str]) -> ClauseClassifier:
        self.model.fit(texts, labels)
        self.labels = list(self.model.classes_)
        return self

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        probs = self.model.predict_proba(texts)
        out = []
        for row in probs:
            i = int(np.argmax(row))
            out.append(
                Prediction(
                    label=self.labels[i],
                    confidence=float(row[i]),
                    scores={
                        lab: round(float(p), 4) for lab, p in zip(self.labels, row, strict=True)
                    },
                )
            )
        return out


@lru_cache(maxsize=1)
def default_classifier() -> ClauseClassifier:
    texts, labels = load_training()
    return ClauseClassifier().fit(texts, labels)
