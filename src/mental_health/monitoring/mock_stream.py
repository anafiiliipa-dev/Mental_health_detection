"""
Simule l'arrivée de nouveaux messages entrants pour le monitoring du drift.

Le projet n'a pas encore de trafic de production réel, donc — selon la
décision documentée (nœuds 02/15/16 du diagramme d'architecture) — le "mock
stream" est un échantillon tiré du split de test mis de côté du jeu de
données d'entraînement : des lignes sur lesquelles le modèle champion ne
s'est jamais entraîné, faisant office de substitut à de véritables nouveaux
messages jusqu'à ce que de vrais logs de requêtes existent pour remplacer
cela.

Reproduit exactement le même split stratifié que ``build_splits`` de
``train.py`` (même ``TEST_SIZE``, même ``RANDOM_STATE``), afin que
l'ensemble de "référence" ici corresponde toujours à ce sur quoi le modèle
actuellement en service s'est réellement entraîné.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mental_health.data.cleaning import TARGET_COL, TEXT_COL
from mental_health.train.model_registry import RANDOM_STATE

TEST_SIZE = 0.2


def build_reference_and_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Découpe ``df`` exactement comme le fait ``train.py``, en ne conservant que
    les colonnes dont le drift check a besoin (``TEXT_COL``, ``TARGET_COL``).

    Retourne ``(reference_df, holdout_df)`` :
    - ``reference_df`` : les lignes d'entraînement — ce que le modèle a "vu".
    - ``holdout_df`` : les lignes de test intactes — le pool d'échantillonnage
      pour le mock stream.
    """
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=df[TARGET_COL], random_state=RANDOM_STATE
    )
    reference_df = df.loc[train_idx, [TEXT_COL, TARGET_COL]].reset_index(drop=True)
    holdout_df = df.loc[test_idx, [TEXT_COL, TARGET_COL]].reset_index(drop=True)
    return reference_df, holdout_df


def sample_mock_batch(
    holdout_df: pd.DataFrame, n: int, random_state: int | None = None, simulate_drift: bool = False
) -> pd.DataFrame:
    """
    Échantillonne ``n`` lignes du pool holdout — un lot simulé de messages
    nouvellement arrivés.

    ``random_state=None`` (la valeur par défaut) tire un échantillon
    différent à chaque appel, ce qu'un vrai run cron hebdomadaire devrait
    faire. Passer une graine fixe dans les tests pour la reproductibilité.
    Si le pool a moins de ``n`` lignes, le pool entier est retourné au lieu
    de lever une erreur.

    ``simulate_drift=True`` retourne volontairement un lot BIAISÉ plutôt
    qu'un échantillon aléatoire honnête — décidé avec Ana : sans trafic de
    production réel pour l'instant, un échantillon holdout honnête est tiré
    exactement de la même distribution que l'entraînement, donc il ne
    déclenche pratiquement jamais le preset de drift d'Evidently. Cela rend
    toute la boucle détection -> alerte -> réentraînement impossible à
    observer de bout en bout. Ce mode existe uniquement pour exercer cette
    boucle sur une cadence prévisible (voir drift_monitoring.yml, qui
    s'exécute désormais toujours avec cette option activée) -- ce n'est PAS
    une simulation de trafic réaliste et ne doit pas être interprété comme
    tel.

    Le biais touche les deux colonnes qu'Evidently compare dans
    build_drift_frames :
    - "prediction" (drift catégoriel) : le lot est tiré d'un seul label
      (le plus rare du pool), au lieu du mélange de classes naturel du pool.
    - "text_length" (drift numérique) : chaque texte échantillonné est
      dupliqué avec lui-même, doublant environ sa longueur.
    """
    if not simulate_drift:
        n = min(n, len(holdout_df))
        return holdout_df.sample(n=n, random_state=random_state).reset_index(drop=True)

    skewed_label = holdout_df[TARGET_COL].value_counts().idxmin()
    skewed_pool = holdout_df[holdout_df[TARGET_COL] == skewed_label]
    batch = skewed_pool.sample(n=n, random_state=random_state, replace=True).reset_index(drop=True)
    batch[TEXT_COL] = batch[TEXT_COL] + " " + batch[TEXT_COL]
    return batch
