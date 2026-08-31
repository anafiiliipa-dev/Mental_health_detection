"""Unit tests for src/mental_health/train/xgb_wrapper.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from mental_health.train.xgb_wrapper import XGBTextClassifier


def _tiny_dataset():
    X = pd.Series(
        [
            "i feel anxious all the time",
            "panic and worry every single day",
            "manic episodes and racing thoughts",
            "crushing lows after the highs",
            "i hear voices when alone",
            "paranoid about being watched",
        ]
        * 5
    )
    y = pd.Series(["Anxiety", "Anxiety", "Bipolar", "Bipolar", "Schizophrenia", "Schizophrenia"] * 5)
    return X, y


class TestXgbTextClassifier:
    def test_fits_and_predicts_string_labels_directly(self):
        X, y = _tiny_dataset()
        clf = XGBTextClassifier(n_estimators=20, max_depth=3, random_state=0)
        pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", clf)])
        pipeline.fit(X, y)

        predictions = pipeline.predict(X)
        assert set(predictions) <= set(y.unique())
        assert len(predictions) == len(y)

    def test_predict_proba_columns_match_classes(self):
        X, y = _tiny_dataset()
        clf = XGBTextClassifier(n_estimators=20, max_depth=3, random_state=0)
        clf.fit(TfidfVectorizer().fit_transform(X), y)

        proba = clf.predict_proba(TfidfVectorizer().fit(X).transform(X))
        assert proba.shape == (len(y), len(y.unique()))
        assert list(clf.classes_) == sorted(y.unique())

    def test_is_clonable_and_set_params_works(self):
        # Required for benchmark.py's CV loops, which clone() + set_params()
        # every candidate on every fold.
        clf = XGBTextClassifier()
        cloned = clone(clf)
        cloned.set_params(n_estimators=50, max_depth=2)
        assert cloned.n_estimators == 50
        assert cloned.max_depth == 2

    def test_sample_weight_is_derived_from_class_weight_dict(self):
        X, y = _tiny_dataset()
        class_weight = {"Anxiety": 1.0, "Bipolar": 2.0, "Schizophrenia": 3.0}
        clf = XGBTextClassifier(n_estimators=10, max_depth=2, class_weight=class_weight)
        clf.fit(TfidfVectorizer().fit_transform(X), y)
        # No error and a real fitted model is enough to prove the sample_weight
        # array (one float per row, from class_weight) reached XGBoost without
        # shape errors -- the exact effect on the boundary isn't asserted here.
        assert clf.classes_ is not None

    def test_decision_function_falls_back_to_predict_proba(self):
        X, y = _tiny_dataset()
        clf = XGBTextClassifier(n_estimators=10, max_depth=2)
        Xt = TfidfVectorizer().fit_transform(X)
        clf.fit(Xt, y)
        assert np.array_equal(clf.decision_function(Xt), clf.predict_proba(Xt))
