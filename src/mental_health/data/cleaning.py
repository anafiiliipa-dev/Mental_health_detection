"""
Pipeline de nettoyage des données pour le jeu de données brut Mental Health Detection.

Extrait et corrigé à partir de ``notebooks/01_data_cleaning.ipynb`` afin que la
logique de nettoyage soit testable, réutilisable et reproductible en ligne de
commande plutôt que de vivre uniquement dans un notebook.

Corrections apportées par rapport au notebook original (voir
``audit-dataset-brut.md`` dans le projet pour l'enquête complète) :

1. ``schizophrenia`` est désormais correctement canonicalisé en ``"Schizophrenia"``
   (title case, cohérent avec tous les autres labels) au lieu de rester en
   minuscules — le mapping original cassait silencieusement les filtres
   sensibles à la casse en aval (par ex. ``CRITICAL_LABELS`` dans le notebook
   d'évaluation clinique).
2. Les doublons exacts (``body`` + ``category`` identiques) sont désormais
   supprimés. Le notebook original n'explorait cela que dans le brouillon
   jetable ``00_exploration.ipynb`` — cela n'a jamais atteint le vrai
   pipeline de nettoyage.
3. Les lignes où le même texte ``body`` apparaît sous plus d'une ``category``
   distincte (bruit d'étiquetage) sont entièrement supprimées, car nous
   n'avons aucun moyen fiable d'arbitrer quel label est correct.
4. Les textes quasi vides (moins de ``MIN_BODY_LENGTH`` caractères) sont
   filtrés en tant que bruit. Les textes longs sont volontairement conservés
   — ce sont des posts verbeux mais légitimes, pas du bruit.
5. La colonne ``body_masked`` (les termes de diagnostic/médication remplacés
   par ``[CONDITION]``, utilisée comme variante textuelle robuste aux fuites
   pour l'entraînement) est ajoutée à la fin du pipeline, via
   ``mental_health.data.masking``. Comportement identique à la logique de
   masquage du notebook — seule la correction de casse des labels ci-dessus
   change quelle variante de texte finit par être sélectionnée comme
   championne en aval.

La détection de quasi-doublons (MinHash/LSH) est volontairement NON incluse
ici. C'est un travail réel et utile déjà prototypé dans
``00_exploration.ipynb``, mais cela ajoute une nouvelle dépendance
(``datasketch``) et une complexité non négligeable — c'est laissé comme
suite documentée plutôt que porté aveuglément.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mental_health.config.paths import DEFAULT_CLEAN_DATA_PATH, RAW_DATA_PATH
from mental_health.data.masking import add_masked_column

logger = logging.getLogger(__name__)

# ============================================================
# Constantes
# ============================================================

TEXT_COL = "body"
TARGET_COL = "category"
MASKED_COL = "body_masked"
REQUIRED_COLUMNS = [TEXT_COL, TARGET_COL]

# Longueur minimale du texte (caractères, après nettoyage) pour conserver une ligne.
# Tout ce qui est plus court est traité comme du bruit, pas comme un post court légitime.
MIN_BODY_LENGTH = 10

# Mapping canonique des labels. Chaque variante brute (casse, synonyme, abréviation)
# est mappée vers exactement un des 7 CLASS_LABELS officiels, tous en title case.
# NOTE : "schizophrenia" est mappé vers "Schizophrenia" (title case) — c'était la
# seule entrée restée en minuscules dans le notebook original.
CANONICAL_LABELS: dict[str, str] = {
    "adhd": "ADHD",
    "add": "ADHD",
    "anxiety": "Anxiety",
    "autism": "Autism",
    "asd": "Autism",
    "autistic": "Autism",
    "bipolar": "Bipolar",
    "bpd": "BPD",
    "borderline": "BPD",
    "depression": "Depression",
    "depressed": "Depression",
    "schizophrenia": "Schizophrenia",
    "schizo": "Schizophrenia",
    "schizoaffective": "Schizophrenia",
}


# ============================================================
# Chargement & validation du schéma
# ============================================================

def load_raw_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Charge le jeu de données brut et valide que les colonnes requises existent.

    Raises
    ------
    FileNotFoundError
        Si ``path`` n'existe pas.
    ValueError
        Si ``body`` ou ``category`` est absent des colonnes.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")

    df = pd.read_csv(path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in raw dataset: {missing_cols}")

    logger.info("Loaded raw dataset: %d rows from %s", len(df), path)
    return df


# ============================================================
# Étapes de nettoyage individuelles
# ============================================================

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Mappe les valeurs brutes de category vers l'ensemble canonique de labels, en supprimant les lignes non mappées."""
    before = len(df)

    df = df.copy()
    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(CANONICAL_LABELS)
    )
    df = df.dropna(subset=[TARGET_COL]).copy()

    removed = before - len(df)
    logger.info("normalize_labels: removed %d unmapped/empty-label rows (%d -> %d)", removed, before, len(df))
    return df


def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes dont le texte ou le label est manquant."""
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    removed = before - len(df)
    logger.info("drop_missing_values: removed %d rows (%d -> %d)", removed, before, len(df))
    return df


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes qui sont des doublons exacts de (body, category)."""
    before = len(df)
    df = df.drop_duplicates(subset=REQUIRED_COLUMNS).copy()
    removed = before - len(df)
    logger.info("drop_exact_duplicates: removed %d rows (%d -> %d)", removed, before, len(df))
    return df


def drop_label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime chaque ligne dont le texte est associé à plus d'un label distinct.

    Nous n'avons aucun signal fiable pour arbitrer quel label est correct
    dans ces cas, donc les occurrences en conflit (deux ou plus) sont
    toutes supprimées plutôt que devinées.
    """
    before = len(df)

    label_counts = df.groupby(TEXT_COL)[TARGET_COL].nunique()
    conflicting_bodies = label_counts[label_counts > 1].index

    df = df[~df[TEXT_COL].isin(conflicting_bodies)].copy()

    removed = before - len(df)
    logger.info(
        "drop_label_conflicts: removed %d rows across %d conflicting texts (%d -> %d)",
        removed, len(conflicting_bodies), before, len(df),
    )
    return df


def filter_short_texts(df: pd.DataFrame, min_length: int = MIN_BODY_LENGTH) -> pd.DataFrame:
    """Supprime les lignes dont le texte (nettoyé) est plus court que ``min_length`` caractères."""
    before = len(df)
    text_len = df[TEXT_COL].astype(str).str.strip().str.len()
    df = df[text_len >= min_length].copy()
    removed = before - len(df)
    logger.info("filter_short_texts: removed %d rows shorter than %d chars (%d -> %d)", removed, min_length, before, len(df))
    return df


# ============================================================
# Orchestrateur
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Exécute le pipeline de nettoyage complet dans l'ordre et retourne le dataframe nettoyé."""
    logger.info("Starting cleaning pipeline: %d raw rows", len(df))

    df = normalize_labels(df)
    df = drop_missing_values(df)
    df = drop_exact_duplicates(df)
    df = drop_label_conflicts(df)
    df = filter_short_texts(df)
    df = add_masked_column(df, text_col=TEXT_COL, masked_col=MASKED_COL)

    df = df.reset_index(drop=True)
    logger.info("Cleaning pipeline complete: %d rows remaining", len(df))
    return df


def run(input_path: Path = RAW_DATA_PATH, output_path: Path = DEFAULT_CLEAN_DATA_PATH) -> pd.DataFrame:
    """Charge, nettoie et exporte le jeu de données. Retourne le dataframe nettoyé."""
    df_raw = load_raw_dataset(input_path)
    df_clean = clean_dataset(df_raw)

    # Ordre de colonnes fixe (body, body_masked, category), correspondant au
    # format d'export du notebook original.
    df_clean = df_clean[[TEXT_COL, MASKED_COL, TARGET_COL]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    logger.info("Saved cleaned dataset to %s (%d rows)", output_path, len(df_clean))
    return df_clean


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
