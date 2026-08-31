"""
Thin sklearn-compatible wrapper around ``xgboost.XGBClassifier`` (Phase 11:
publishing XGBoost/LightGBM results on TF-IDF alongside the classical
linear registry).

Why this exists: xgboost>=2's sklearn API requires integer-encoded targets
(0..K-1) and raises on raw string class labels -- unlike every other
classifier in ``model_registry.py`` (LinearSVC, LogisticRegression,
MultinomialNB, and LightGBM's own sklearn API), which all accept the
project's string labels directly. This wrapper label-encodes at fit time
and decodes predictions back to the original strings, so XGBoost drops
into the existing pipelines/benchmark/champion code exactly like every
other candidate -- nothing in benchmark.py, champion.py or train.py needs
to know XGBoost is different.

Also translates this project's existing per-class weight dict convention
(``model_registry.compute_boosted_class_weights``, the same one LinearSVC/
LogisticRegression already take as ``class_weight=``) into XGBoost's
``sample_weight`` fit argument, since ``XGBClassifier`` has no
``class_weight`` parameter of its own.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class XGBTextClassifier(BaseEstimator, ClassifierMixin):
    """XGBClassifier that accepts string labels and an optional per-class weight dict, like the rest of the registry."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        class_weight: dict[str, float] | None = None,
    ):
        # Every constructor arg must be stored verbatim as an identically-
        # named attribute, unmodified -- sklearn's get_params()/clone()
        # contract (used throughout benchmark.py's CV loops) depends on it.
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.class_weight = class_weight

    def fit(self, X, y):  # noqa: N803
        self._label_encoder = LabelEncoder().fit(y)
        self.classes_ = self._label_encoder.classes_

        sample_weight = None
        if self.class_weight is not None:
            sample_weight = np.array([self.class_weight[label] for label in y])

        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            eval_metric="mlogloss",
        )
        self._model.fit(X, self._label_encoder.transform(y), sample_weight=sample_weight)
        return self

    def predict(self, X):  # noqa: N803
        return self._label_encoder.inverse_transform(self._model.predict(X))

    def predict_proba(self, X):  # noqa: N803
        # XGBClassifier's column order follows the label encoder's fit
        # order, which is exactly self.classes_ (both alphabetically
        # sorted) -- no reordering needed.
        return self._model.predict_proba(X)

    def decision_function(self, X):  # noqa: N803
        # Not natively provided by XGBClassifier -- exposed anyway so any
        # caller that prefers decision_function over predict_proba (none
        # currently do; get_ranking_scores in evaluation_metrics.py checks
        # predict_proba first) still gets ranking scores instead of an
        # AttributeError.
        return self.predict_proba(X)
