"""
Prédiction de secours déterministe, basée sur des mots-clés — utilisée UNIQUEMENT lorsqu'aucun
modèle réel n'est disponible (voir ``model_loader.load_production_model``).

Ceci n'est explicitement PAS un modèle clinique. Elle existe pour que l'API reste utilisable
en mode dégradé (par exemple en développement local avant qu'un modèle ne soit promu, ou lors d'une
panne du Registry) plutôt que de faire échouer chaque requête. ``main.py`` marque toujours
les réponses produites par ce module avec ``is_demo_fallback=True`` afin qu'aucun
appelant ne puisse la confondre avec une véritable prédiction.
"""
from __future__ import annotations

from mental_health.config.paths import CLASS_LABELS

# Ensembles de mots-clés petits et illustratifs — suffisants pour que le mode démo se comporte
# de façon raisonnable, mais ne remplacent pas le modèle entraîné. Volontairement simples.
_KEYWORDS: dict[str, list[str]] = {
    "ADHD": ["adhd", "hyperactive", "hyperactivity", "distracted", "can't focus", "inattentive"],
    "Anxiety": ["anxious", "anxiety", "panic", "worry", "worried", "on edge"],
    "Autism": ["autism", "autistic", "spectrum", "asperger"],
    "Bipolar": ["bipolar", "manic", "mania", "mood swings"],
    "BPD": ["borderline", "bpd", "abandonment"],
    "Depression": ["depress", "hopeless", "empty", "worthless", "no motivation"],
    "Schizophrenia": ["schizophrenia", "psychosis", "hallucinat", "hearing voices", "paranoid"],
}

# Volontairement bas, afin qu'un modèle réel ne soit jamais surpassé sur le papier par le fallback.
_MATCH_CONFIDENCE = 0.35
_NO_MATCH_CONFIDENCE = 1.0 / len(CLASS_LABELS)


def fallback_demo_prediction(text: str) -> tuple[str, float, dict[str, float]]:
    """
    Heuristique très simple de comptage de mots-clés sur ``CLASS_LABELS``.

    Retourne (label, confidence, probabilities) avec la même forme que la
    sortie du modèle réel, afin que ``main.py`` puisse construire une ``PredictResponse``
    de façon identique dans les deux cas.
    """
    lowered = text.lower()
    scores = {label: sum(lowered.count(kw) for kw in keywords) for label, keywords in _KEYWORDS.items()}
    total_hits = sum(scores.values())

    if total_hits == 0:
        probabilities = dict.fromkeys(CLASS_LABELS, _NO_MATCH_CONFIDENCE)
        return "Depression", _NO_MATCH_CONFIDENCE, probabilities

    label = max(scores, key=scores.get)
    probabilities = {lbl: round(count / total_hits, 4) for lbl, count in scores.items()}
    confidence = max(probabilities[label], _MATCH_CONFIDENCE)
    return label, confidence, probabilities
