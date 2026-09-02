"""
Wrapper léger et compatible sklearn autour de ``xgboost.XGBClassifier`` (Phase 11 :
publication des résultats XGBoost/LightGBM sur TF-IDF aux côtés du
registre linéaire classique).

Pourquoi cela existe : l'API sklearn de xgboost>=2 nécessite des cibles encodées en
entiers (0..K-1) et lève une exception sur des labels de classe en chaîne brute -- contrairement à
tout autre classifieur dans ``model_registry.py`` (LinearSVC, LogisticRegression,
MultinomialNB, et l'API sklearn propre à LightGBM), qui acceptent tous
directement les labels en chaîne du projet. Ce wrapper encode les labels au moment du
fit et décode les prédictions vers les chaînes d'origine, afin que XGBoost s'insère
dans les pipelines/benchmark/champion existants exactement comme tout
autre candidat -- rien dans benchmark.py, champion.py ou train.py n'a besoin
de savoir que XGBoost est différent.

Traduit aussi la convention existante de ce projet pour le dict de poids par
classe (``model_registry.compute_boosted_class_weights``, le même que LinearSVC/
LogisticRegression prennent déjà comme ``class_weight=``) vers l'argument
``sample_weight`` de fit de XGBoost, puisque ``XGBClassifier`` n'a pas de
paramètre ``class_weight`` propre.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class XGBTextClassifier(BaseEstimator, ClassifierMixin):
    """XGBClassifier qui accepte des labels en chaîne et un dict optionnel de poids par classe, comme le reste du registre."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        class_weight: dict[str, float] | None = None,
    ):
        # Chaque argument du constructeur doit être stocké tel quel comme un
        # attribut de même nom, non modifié -- le contrat get_params()/clone()
        # de sklearn (utilisé partout dans les boucles CV de benchmark.py) en dépend.
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
        # L'ordre des colonnes de XGBClassifier suit l'ordre de fit du label
        # encoder, qui est exactement self.classes_ (les deux triés par
        # ordre alphabétique) -- aucune réorganisation nécessaire.
        return self._model.predict_proba(X)

    def decision_function(self, X):  # noqa: N803
        # Non fourni nativement par XGBClassifier -- exposé quand même pour que tout
        # appelant qui préfère decision_function à predict_proba (aucun ne
        # le fait actuellement ; get_ranking_scores dans evaluation_metrics.py vérifie
        # predict_proba en premier) obtienne quand même des scores de classement plutôt qu'une
        # AttributeError.
        return self.predict_proba(X)
