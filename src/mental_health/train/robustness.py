"""
Tests de robustesse sous perturbation textuelle (Phase 11, tranche
restante : "tests de robustesse (fautes de frappe/casse)").

Évalue un modèle déjà entraîné (typiquement le champion enregistré) sur
des copies délibérément corrompues de l'ensemble de test mis de côté — fautes de frappe et
changements de casse qu'un vrai utilisateur est susceptible de réellement taper — et rapporte de combien
chaque métrique principale se dégrade par rapport à la baseline propre. Ceci est
uniquement du reporting/diagnostic, exactement comme le MCC/PR-AUC et le test bootstrap
apparié d'``evaluation_metrics.py`` : cela n'influence PAS en retour la sélection
du champion (``benchmark.py``/``champion.py``) ni la porte de promotion
(``promote.py``), qui restent toutes deux inchangées.

Les perturbations sont appliquées à du texte déjà nettoyé (``cleaning.py`` a
déjà tourné), simulant une entrée bruitée au moment de l'inférence — pas des
données d'entraînement bruitées — puisque les pipelines TF-IDF/embedding de
``build_model_registry`` sont ajustés une seule fois sur du texte propre et jamais
réentraînés ici ; seul ``X_test`` est corrompu avant d'appeler ``model.predict``.
"""
from __future__ import annotations

import random
import string
from functools import partial

import pandas as pd
from sklearn.metrics import f1_score, recall_score

from mental_health.train.benchmark import critical_recall_score

# ============================================================
# Perturbations
# ============================================================


def inject_typos(text: str, rate: float = 0.1, seed: int | None = None) -> str:
    """
    Corrompt ``text`` caractère par caractère : chaque caractère a
    indépendamment une probabilité ``rate`` d'être altéré par l'une des quatre
    erreurs de frappe courantes (substitution par une lettre minuscule aléatoire, suppression,
    duplication, ou transposition avec le caractère suivant). Déterministe pour une
    ``seed`` donnée afin que le rapport soit reproductible.
    """
    if not text:
        return text

    rng = random.Random(seed)
    chars = list(text)
    out = []
    i = 0
    while i < len(chars):
        char = chars[i]
        if char.isalpha() and rng.random() < rate:
            mistake = rng.choice(["substitute", "delete", "duplicate", "transpose"])
            if mistake == "substitute":
                out.append(rng.choice(string.ascii_lowercase))
            elif mistake == "delete":
                pass  # supprime le caractère entièrement
            elif mistake == "duplicate":
                out.append(char)
                out.append(char)
            elif mistake == "transpose" and i + 1 < len(chars):
                out.append(chars[i + 1])
                out.append(char)
                i += 1  # a déjà consommé le caractère suivant
        else:
            out.append(char)
        i += 1

    return "".join(out)


def randomize_casing(text: str, mode: str = "random", seed: int | None = None) -> str:
    """
    Change la casse de ``text``. ``mode`` vaut l'un des suivants :

    - ``"upper"`` : TOUT EN MAJUSCULES (par ex. un utilisateur avec le verrouillage majuscules bloqué).
    - ``"lower"`` : tout en minuscules (le cas réel le plus courant — la plupart
      des gens ne se donnent pas la peine de mettre des majuscules dans un texte informel).
    - ``"random"`` : chaque lettre indépendamment mise en majuscule/minuscule — un
      cas extrême de test de stress, peu probable tel quel mais qui exerce la même
      sensibilité à la casse dont un vrai modèle ne devrait pas dépendre.
    """
    if not text:
        return text

    if mode == "upper":
        return text.upper()
    if mode == "lower":
        return text.lower()
    if mode == "random":
        rng = random.Random(seed)
        return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in text)

    raise ValueError(f"Unknown casing mode: {mode!r}")


def perturb_series(texts: pd.Series, perturb_fn, random_state: int = 42) -> pd.Series:
    """
    Applique ``perturb_fn(text, seed=...)`` à chaque ligne de ``texts``, en initialisant
    chaque ligne de manière déterministe à partir de sa position afin que relancer la même
    perturbation sur les mêmes données produise toujours le même texte corrompu.
    """
    texts = pd.Series(texts).reset_index(drop=True)
    return pd.Series([perturb_fn(text, seed=random_state + i) for i, text in enumerate(texts)])


# Presets exercés par défaut par evaluate_robustness / run(). Chacun associe un
# nom lisible par un humain à une fonction de perturbation prête à l'emploi
# à un argument près (a encore besoin de ``seed=`` au moment de l'appel, fourni par perturb_series).
DEFAULT_PERTURBATIONS = {
    "typos_light": partial(inject_typos, rate=0.05),
    "typos_heavy": partial(inject_typos, rate=0.15),
    "uppercase": partial(randomize_casing, mode="upper"),
    "lowercase": partial(randomize_casing, mode="lower"),
    "random_case": partial(randomize_casing, mode="random"),
}


# ============================================================
# Évaluation
# ============================================================


def _compute_metrics(y_true, y_pred) -> dict:
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "critical_recall": critical_recall_score(y_true, y_pred),
    }


def evaluate_robustness(
    model,
    X_test,
    y_test,
    perturbations: dict | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Évalue ``model`` sur l'ensemble de test propre, puis sur chaque copie perturbée
    de ``perturbations`` (par défaut ``DEFAULT_PERTURBATIONS``). Retourne
    une ligne par condition (``"clean"`` en premier) avec les trois métriques
    principales plus leur delta par rapport à la baseline propre — un delta négatif
    signifie que la perturbation a nui à la performance.
    """
    if perturbations is None:
        perturbations = DEFAULT_PERTURBATIONS

    X_test = pd.Series(X_test).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)

    baseline_pred = model.predict(X_test)
    baseline_metrics = _compute_metrics(y_test, baseline_pred)

    rows = [{
        "perturbation": "clean",
        **baseline_metrics,
        "delta_f1_macro": 0.0,
        "delta_recall_macro": 0.0,
        "delta_critical_recall": 0.0,
    }]

    for name, perturb_fn in perturbations.items():
        X_perturbed = perturb_series(X_test, perturb_fn, random_state=random_state)
        y_pred = model.predict(X_perturbed)
        metrics = _compute_metrics(y_test, y_pred)
        rows.append({
            "perturbation": name,
            **metrics,
            "delta_f1_macro": metrics["f1_macro"] - baseline_metrics["f1_macro"],
            "delta_recall_macro": metrics["recall_macro"] - baseline_metrics["recall_macro"],
            "delta_critical_recall": metrics["critical_recall"] - baseline_metrics["critical_recall"],
        })

    return pd.DataFrame(rows)


def summarize_worst_case(report: pd.DataFrame) -> dict:
    """
    Pire dégradation parmi toutes les perturbations non "clean" d'un
    rapport ``evaluate_robustness`` — le chiffre unique à surveiller : "à quel point
    ça peut être mauvais sur une entrée réaliste et désordonnée ?".
    """
    perturbed = report[report["perturbation"] != "clean"]
    if perturbed.empty:
        return {"worst_delta_f1_macro": 0.0, "worst_delta_critical_recall": 0.0, "worst_perturbation": None}

    worst_row = perturbed.loc[perturbed["delta_f1_macro"].idxmin()]
    return {
        "worst_delta_f1_macro": float(perturbed["delta_f1_macro"].min()),
        "worst_delta_critical_recall": float(perturbed["delta_critical_recall"].min()),
        "worst_perturbation": str(worst_row["perturbation"]),
    }


def run() -> pd.DataFrame:
    """
    Charge le champion actuel (alias "production" enregistré, avec repli
    sur "staging" si rien n'est encore promu), reconstruit le même
    split train/test que celui utilisé pour l'entraîner, exécute ``evaluate_robustness`` sur
    l'ensemble de test de la variante de texte "raw", écrit le rapport dans
    ``paths.ROBUSTNESS_REPORT_PATH`` et le retourne.
    """
    import logging

    import mlflow
    from dotenv import load_dotenv

    load_dotenv()

    from mental_health.config.mlflow_config import (
        MLFLOW_REGISTERED_MODEL_NAME,
        MLFLOW_TRACKING_URI,
        PRODUCTION_ALIAS,
        STAGING_ALIAS,
    )
    from mental_health.config.paths import DEFAULT_CLEAN_DATA_PATH, ROBUSTNESS_REPORT_PATH
    from mental_health.train.train import build_splits

    logger = logging.getLogger(__name__)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    try:
        version = client.get_model_version_by_alias(MLFLOW_REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
        alias_used = PRODUCTION_ALIAS
    except mlflow.exceptions.MlflowException:
        version = client.get_model_version_by_alias(MLFLOW_REGISTERED_MODEL_NAME, STAGING_ALIAS)
        alias_used = STAGING_ALIAS

    model = mlflow.sklearn.load_model(f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@{alias_used}")
    logger.info("Loaded '%s' v%s (alias=%s) for robustness evaluation", MLFLOW_REGISTERED_MODEL_NAME, version.version, alias_used)

    df = pd.read_csv(DEFAULT_CLEAN_DATA_PATH)
    splits = build_splits(df)
    X_test, y_test = splits["raw"]["X_test"], splits["raw"]["y_test"]

    report = evaluate_robustness(model, X_test, y_test)
    worst_case = summarize_worst_case(report)
    logger.info("Robustness worst case: %s", worst_case)

    ROBUSTNESS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(ROBUSTNESS_REPORT_PATH, index=False)
    logger.info("Robustness report written to %s:\n%s", ROBUSTNESS_REPORT_PATH, report.to_string(index=False))

    return report


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
