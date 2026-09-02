"""
Job hebdomadaire de monitoring du drift (fin de la Phase 10).

Score un lot échantillonné de "nouveaux" messages (voir ``mock_stream.py``) avec le
modèle "production" actuel, compare la distribution des labels prédits
et de la longueur du texte par rapport au jeu de référence d'entraînement en
utilisant Evidently, et :

- écrit un rapport de drift HTML (téléversé en tant qu'artefact de run
  GitHub Actions par le workflow appelant),
- consigne un run récapitulatif (métriques de drift + le rapport lui-même)
  dans le backend MLflow partagé, sous l'expérience ``mental_health_monitoring``.

Ne déclenche volontairement PAS de réentraînement — cette décision est
explicitement différée à la Phase 12 (déclenchement automatique du
réentraînement). Ce script se contente d'observer et de rapporter.

Sensible au flavor (ajouté quand model_loader/main.py de l'API est devenu
sensible au flavor — voir ``mental_health.api.main._predict_with_real_model``) :
le modèle "production" peut désormais être un pipeline scikit-learn OU un
pipeline Transformers de classification de texte (par ex. un candidat
DistilBERT promu), et ce job score celui qui est réellement en production
au lieu de supposer ``.predict``/``.predict_proba`` — voir
``_predict_batch`` ci-dessous, qui reproduit la même logique de dispatch
que l'API utilise par requête, ici traitée par lot pour scorer un dataframe
entier d'un coup.

Usage:
    python -m mental_health.monitoring.drift_check
    python -m mental_health.monitoring.drift_check --batch-size 100 --report-path drift.html
"""
from __future__ import annotations

import argparse
import logging
import os

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Doit s'exécuter avant que l'import mlflow_config ci-dessous ne lise MLFLOW_TRACKING_URI /
# MLFLOW_ARTIFACT_ROOT depuis l'environnement — même convention que
# train.py / promote.py / main.py. Un no-op en CI, où ces variables sont définies
# directement comme secrets/variables d'environnement GitHub Actions.
load_dotenv()

from evidently.metric_preset import DataDriftPreset  # noqa: E402
from evidently.report import Report  # noqa: E402

from mental_health.api.model_loader import LoadedModel, load_production_model  # noqa: E402
from mental_health.config.mlflow_config import (  # noqa: E402
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_TRACKING_URI,
)
from mental_health.config.paths import DEFAULT_CLEAN_DATA_PATH  # noqa: E402
from mental_health.data.cleaning import TEXT_COL  # noqa: E402
from mental_health.monitoring.mock_stream import (  # noqa: E402
    build_reference_and_holdout,
    sample_mock_batch,
)

logger = logging.getLogger(__name__)

MONITORING_EXPERIMENT_NAME = "mental_health_monitoring"
DEFAULT_BATCH_SIZE = 50


def _get_or_create_experiment(name: str, artifact_location: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name, artifact_location=artifact_location)


def _predict_batch(loaded: LoadedModel, texts: pd.Series) -> tuple[list[str], list[float] | None]:
    """
    Score un lot de textes avec le modèle de production, en dispatchant selon
    le flavor exactement comme le fait ``mental_health.api.main._predict_with_real_model``
    par requête. Retourne ``(predicted_labels, confidences)`` --
    ``confidences`` n'est ``None`` que pour un modèle sklearn n'ayant ni
    ``predict_proba`` ni ``decision_function`` (défensif uniquement : depuis
    la Phase 11 le champion classique servi est toujours calibré via
    CalibratedClassifierCV, donc cela devrait toujours être disponible pour ce
    flavor en pratique).
    """
    texts = list(texts)

    if loaded.flavor == "transformers":
        # top_k=None et truncation=True explicitement, à chaque appel -- voir
        # la docstring de _predict_with_transformers_model dans main.py pour savoir
        # pourquoi on ne peut se fier à aucun des deux comme valeur par défaut propre
        # au pipeline après un aller-retour (et pourquoi un texte long non tronqué
        # fait planter les position embeddings de PyTorch au lieu d'échouer proprement).
        results = loaded.model(texts, top_k=None, truncation=True)
        labels: list[str] = []
        confidences: list[float] = []
        for scores in results:
            best = max(scores, key=lambda item: item["score"])
            labels.append(str(best["label"]))
            confidences.append(float(best["score"]))
        return labels, confidences

    model = loaded.model
    labels = [str(p) for p in model.predict(texts)]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)
        return labels, [float(row.max()) for row in proba]

    if hasattr(model, "decision_function"):
        raw_scores = np.atleast_2d(model.decision_function(texts))
        confidences = []
        for row in raw_scores:
            exp_row = np.exp(row - np.max(row))
            confidences.append(float(np.max(exp_row / exp_row.sum())))
        return labels, confidences

    return labels, None


def build_drift_frames(
    loaded: LoadedModel, reference_df: pd.DataFrame, current_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit les colonnes qu'Evidently compare entre le jeu de référence et le
    lot échantillonné actuel : ``text_length`` et ``prediction`` (drift
    catégoriel/cible) toujours, plus ``prediction_confidence`` (drift
    numérique) chaque fois que le modèle de production expose un score de
    confiance pour son flavor.
    """
    reference_labels, reference_confidence = _predict_batch(loaded, reference_df[TEXT_COL])
    current_labels, current_confidence = _predict_batch(loaded, current_df[TEXT_COL])

    reference = pd.DataFrame(
        {
            "text_length": reference_df[TEXT_COL].str.len(),
            "prediction": reference_labels,
        }
    )
    current = pd.DataFrame(
        {
            "text_length": current_df[TEXT_COL].str.len(),
            "prediction": current_labels,
        }
    )

    if reference_confidence is not None and current_confidence is not None:
        reference["prediction_confidence"] = reference_confidence
        current["prediction_confidence"] = current_confidence

    return reference, current


def run_drift_check(
    batch_size: int = DEFAULT_BATCH_SIZE, report_path: str = "drift_report.html", simulate_drift: bool = False
) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    loaded = load_production_model()
    if not loaded.is_available:
        raise SystemExit(f"No production model available — cannot run drift check: {loaded.error}")

    df = pd.read_csv(DEFAULT_CLEAN_DATA_PATH)
    reference_df, holdout_pool = build_reference_and_holdout(df)
    current_df = sample_mock_batch(holdout_pool, n=batch_size, simulate_drift=simulate_drift)

    reference, current = build_drift_frames(loaded, reference_df, current_df)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(report_path)

    drift_result = report.as_dict()["metrics"][0]["result"]
    dataset_drift = bool(drift_result.get("dataset_drift", False))
    n_drifted = int(drift_result.get("number_of_drifted_columns", 0))

    logger.info(
        "Drift check: model v%s (flavor=%s), batch_size=%d, dataset_drift=%s, drifted_columns=%d",
        loaded.version,
        loaded.flavor,
        len(current_df),
        dataset_drift,
        n_drifted,
    )

    experiment_id = _get_or_create_experiment(MONITORING_EXPERIMENT_NAME, MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_param("model_version", loaded.version)
        mlflow.log_param("model_flavor", loaded.flavor)
        mlflow.log_param("batch_size", len(current_df))
        mlflow.log_param("simulate_drift", simulate_drift)
        mlflow.log_metric("dataset_drift", int(dataset_drift))
        mlflow.log_metric("n_drifted_columns", n_drifted)
        mlflow.log_artifact(report_path)

    return {"dataset_drift": dataset_drift, "n_drifted_columns": n_drifted, "report_path": report_path}


def _write_github_output(result: dict) -> None:
    """
    Lorsqu'exécuté dans une étape GitHub Actions, expose le résultat du drift en
    tant que sorties d'étape (``dataset_drift``, ``n_drifted_columns``) afin que le
    workflow appelant puisse décider d'ouvrir ou non une issue d'alerte. Un no-op hors CI.
    """
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as fh:
        fh.write(f"dataset_drift={str(result['dataset_drift']).lower()}\n")
        fh.write(f"n_drifted_columns={result['n_drifted_columns']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the drift-monitoring check against the production model.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--report-path", type=str, default="drift_report.html")
    parser.add_argument(
        "--simulate-drift",
        action="store_true",
        help=(
            "Deliberately sample a skewed batch instead of an honest holdout "
            "sample, so a drift alert reliably fires. For exercising the "
            "detection -> alert -> retrain loop on a predictable cadence "
            "while there is no live production traffic yet -- NOT a "
            "realistic traffic simulation."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = run_drift_check(batch_size=args.batch_size, report_path=args.report_path, simulate_drift=args.simulate_drift)
    logger.info("Drift check complete: %s", result)
    _write_github_output(result)


if __name__ == "__main__":
    main()
