"""
Masquage des termes de diagnostic/médication — produit la colonne ``body_masked``.

Porté depuis ``notebooks/01_data_cleaning.ipynb`` (section d'audit des fuites).
Le raisonnement : un modèle qui peut simplement lire le diagnostic ou le nom
du médicament dans le texte ("I take Lithium for my Bipolar") n'apprend pas
un vrai signal linguistique, il lit la réponse. Masquer ces termes avec un
placeholder neutre donne une variante de texte qui mesure si le modèle
performe toujours bien sans ce raccourci — utilisé comme test de robustesse
en complément du texte brut (voir ``mental_health.train.champion`` pour la
façon dont la comparaison "raw" vs "masked" alimente la sélection du
champion).

Aucun changement de comportement par rapport au notebook — les listes de
termes et la logique regex sont portées telles quelles, seulement
réorganisées en fonctions pures et testables.
"""
from __future__ import annotations

import re

import pandas as pd

MEDICATION_TERMS: list[str] = [
    "ritalin", "concerta", "vyvanse", "adderall", "strattera",
    "lithium", "lamotrigine", "lamictal", "depakote", "valproate",
    "lexapro", "zoloft", "prozac", "sertraline", "escitalopram", "fluoxetine",
    "xanax", "alprazolam", "valium", "diazepam", "ativan", "lorazepam",
    "clozapine", "clozaril", "risperidone", "risperdal", "olanzapine",
    "zyprexa", "quetiapine", "seroquel", "haldol",
]

DIAGNOSTIC_TERMS: list[str] = [
    "adhd", "add", "asd", "autism", "autistic", "bpd", "borderline",
    "bipolar", "schizophrenia", "schizoaffective", "schizo",
    "depression", "depressed", "anxiety", "anxious",
]

FULL_LEAK_LIST: list[str] = DIAGNOSTIC_TERMS + MEDICATION_TERMS

# "add" est spécifiquement exclu de la liste de masquage (conservé dans
# FULL_LEAK_LIST ci-dessus uniquement pour l'*audit* des fuites, pas pour le masquage) :
# c'est un mot anglais courant ("add up", "in addition") et masquer chaque
# occurrence détruirait du texte légitime sans rapport.
MASKING_LEAK_LIST: list[str] = [term for term in FULL_LEAK_LIST if term.lower() != "add"]

MASK_TOKEN = "[CONDITION]"


def build_leakage_pattern(term_list: list[str]) -> str:
    """Construit une regex d'alternation insensible à la casse, sur mots entiers, pour les termes donnés."""
    escaped_terms = [re.escape(term) for term in term_list]
    return rf"(?<!\w)({'|'.join(escaped_terms)})(?!\w)"


LEAKAGE_PATTERN = build_leakage_pattern(MASKING_LEAK_LIST)


def mask_leakage_terms(text: str | float | None) -> str | None:
    """Remplace chaque terme de diagnostic/médication dans ``text`` par ``[CONDITION]``."""
    if pd.isna(text):
        return None
    return re.sub(LEAKAGE_PATTERN, MASK_TOKEN, str(text), flags=re.IGNORECASE)


def add_masked_column(df: pd.DataFrame, text_col: str = "body", masked_col: str = "body_masked") -> pd.DataFrame:
    """Ajoute ``masked_col`` à ``df`` en masquant les termes de diagnostic/médication dans ``text_col``."""
    df = df.copy()
    df[masked_col] = df[text_col].apply(mask_leakage_terms)
    return df
