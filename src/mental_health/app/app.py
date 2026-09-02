from __future__ import annotations

import sys
from html import escape
from pathlib import Path
from typing import Any

# ============================================================
# Configuration des chemins — un seul ajout pour que tous les
# imports src.* se résolvent correctement
# ============================================================

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import mlflow
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

from mental_health.api.main import _predict_with_sklearn_model, _predict_with_transformers_model
from mental_health.api.model_loader import load_production_model
from mental_health.app.openrouter_client import ask_llm, get_default_model
from mental_health.config.mlflow_config import MLFLOW_TRACKING_URI
from mental_health.config.paths import (
    CLASS_LABELS,
    DEFAULT_CLEAN_DATA_PATH,
    FINAL_TEST_METRICS_PATH,
    GLOBAL_CLINICAL_REVIEW_PATH,
    MODEL_CANDIDATES,
    NESTED_CV_SUMMARY_PATH,
    NORMAL_CV_SUMMARY_PATH,
    SAMPLE_OUTPUTS_DIR,
)
from mental_health.models.services import fallback_demo_prediction, load_model, predict_with_model
from mental_health.rag.simple_rag import build_qa_chain

load_dotenv()

# Option ajoutée dans le sélecteur de modèle de la page "Prédictions" pour
# aller chercher directement le modèle aliasé "production" dans le MLflow
# Model Registry (au lieu des fichiers .joblib locaux ci-dessous) -- c'est
# le même modèle que sert l'API FastAPI (mental_health.api.main).
MLFLOW_PRODUCTION_OPTION = "Modèle en production (MLflow)"


# ============================================================
# Constantes au niveau de l'application
# ============================================================

MAX_HISTORY = 8


# ============================================================
# Configuration de la page
# ============================================================

st.set_page_config(
    page_title="Mental Health Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# État de session
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


# ============================================================
# CSS personnalisé
# ============================================================

def load_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(34,211,238,0.08), transparent 25%),
                radial-gradient(circle at top right, rgba(167,139,250,0.08), transparent 20%),
                linear-gradient(135deg, #0b1220 0%, #111827 45%, #0f172a 100%);
            color: #e5e7eb;
        }

        .block-container {
            padding-top: 1.15rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 1450px;
        }

        header, footer, #MainMenu {
            visibility: hidden;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        .main-title {
            font-size: 2.45rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.15rem;
            letter-spacing: -0.03em;
        }

        .subtitle {
            font-size: 1rem;
            color: #94a3b8;
            margin-bottom: 1.6rem;
        }

        .hero-box {
            background: linear-gradient(135deg, rgba(30,41,59,0.92), rgba(15,23,42,0.90));
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 24px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.28);
            margin-bottom: 1.5rem;
        }

        .hero-title {
            font-size: 1.75rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.45rem;
        }

        .hero-text {
            color: #cbd5e1;
            font-size: 1rem;
            line-height: 1.7;
        }

        .metric-card {
            background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,41,59,0.92));
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 22px;
            padding: 22px 20px;
            box-shadow: 0 12px 24px rgba(0,0,0,0.22);
            min-height: 152px;
        }

        .metric-label {
            color: #94a3b8;
            font-size: 0.92rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
        }

        .metric-value {
            color: #f8fafc;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.5rem;
            word-break: break-word;
        }

        .metric-delta {
            font-size: 0.92rem;
            font-weight: 700;
        }

        .accent-cyan { color: #22d3ee; }
        .accent-gold { color: #fbbf24; }
        .accent-green { color: #34d399; }
        .accent-purple { color: #a78bfa; }
        .accent-red { color: #fb7185; }

        .section-box {
            background: rgba(15,23,42,0.76);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 22px;
            padding: 24px;
            margin-top: 1rem;
            box-shadow: 0 10px 20px rgba(0,0,0,0.18);
        }

        .section-title {
            font-size: 1.18rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.8rem;
        }

        .section-text {
            color: #cbd5e1;
            line-height: 1.7;
            font-size: 0.97rem;
        }

        .small-badge {
            display: inline-block;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            color: #e2e8f0;
            background: rgba(255,255,255,0.07);
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
        }

        .result-box {
            background: linear-gradient(135deg, rgba(17,24,39,0.98), rgba(15,23,42,0.96));
            border: 1px solid rgba(34,211,238,0.18);
            border-radius: 22px;
            padding: 24px;
            box-shadow: 0 12px 26px rgba(0,0,0,0.22);
        }

        .result-label {
            color: #94a3b8;
            font-size: 0.92rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }

        .result-value {
            color: #f8fafc;
            font-size: 1.8rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
        }

        .divider-line {
            height: 1px;
            border: none;
            background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.12), rgba(255,255,255,0.02));
            margin: 1.25rem 0;
        }

        .history-card {
            background: rgba(15,23,42,0.78);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 18px;
            padding: 16px;
            margin-bottom: 0.8rem;
        }

        .history-title {
            color: #f8fafc;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .history-meta {
            color: #94a3b8;
            font-size: 0.88rem;
            margin-bottom: 0.45rem;
        }

        .history-text {
            color: #cbd5e1;
            font-size: 0.94rem;
            line-height: 1.55;
            white-space: pre-wrap;
            word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Fonctions utilitaires
# ============================================================

def metric_card(
    label: str,
    value: str,
    delta_text: str,
    accent_class: str = "accent-cyan",
) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{escape(label)}</div>
        <div class="metric-value">{escape(value)}</div>
        <div class="metric-delta {accent_class}">{escape(delta_text)}</div>
    </div>
    """


def render_header() -> None:
    st.markdown(
        '<div class="main-title">🧠 Mental Health Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Tableau de bord NLP de style clinique pour l\'analyse de texte en '
        'santé mentale, le suivi des modèles, la revue des prédictions et une démonstration prête au déploiement.</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "total_texts": None,
        "num_classes": len(CLASS_LABELS),
        "class_names": CLASS_LABELS,
        "dataset_loaded": False,
        "dataset_path": None,
    }

    if DEFAULT_CLEAN_DATA_PATH.exists():
        try:
            df = pd.read_csv(DEFAULT_CLEAN_DATA_PATH)
            info["dataset_loaded"] = True
            info["dataset_path"] = str(DEFAULT_CLEAN_DATA_PATH)
            info["total_texts"] = len(df)
            if "category" in df.columns:
                unique_classes = sorted(df["category"].dropna().astype(str).unique().tolist())
                if unique_classes:
                    info["class_names"] = unique_classes
                    info["num_classes"] = len(unique_classes)
        except Exception:
            pass

    return info


@st.cache_data(show_spinner=False)
def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    primary_path = path
    sample_path = SAMPLE_OUTPUTS_DIR / path.name

    try:
        if primary_path.exists():
            return pd.read_csv(primary_path)

        if sample_path.exists():
            return pd.read_csv(sample_path)

    except Exception as exc:
        st.warning(f"Impossible de charger {path.name} : {exc}")
        return None

    return None


def safe_get_first_value(df: pd.DataFrame | None, candidate_cols: list[str]) -> Any | None:
    if df is None or df.empty:
        return None
    for col in candidate_cols:
        if col in df.columns:
            return df[col].iloc[0]
    return None


def format_metric(value: Any | None, decimals: int = 3) -> str:
    if value is None:
        return "N/D"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


@st.cache_data(show_spinner=False)
def load_monitoring_artifacts() -> dict[str, pd.DataFrame | None]:
    return {
        "final_test_df":  load_csv_if_exists(FINAL_TEST_METRICS_PATH),
        "nested_cv_df":   load_csv_if_exists(NESTED_CV_SUMMARY_PATH),
        "normal_cv_df":   load_csv_if_exists(NORMAL_CV_SUMMARY_PATH),
        "global_review_df": load_csv_if_exists(GLOBAL_CLINICAL_REVIEW_PATH),
    }


@st.cache_resource(show_spinner=False)
def load_joblib_model(model_name: str) -> tuple[Any | None, Path | None, str | None]:
    return load_model(model_name)


@st.cache_resource(show_spinner=False)
def load_mlflow_production_model():
    """
    Charge le modèle actuellement aliasé "production" dans le MLflow Model
    Registry -- le même modèle, servi par le même backend partagé (Neon +
    S3), que l'API FastAPI sert via /predict. Mis en cache par Streamlit
    (st.cache_resource) pour ne pas retélécharger le modèle à chaque clic ;
    utilise "Effacer l'historique des prédictions" puis recharge la page
    pour forcer un rechargement si une nouvelle version a été promue entre
    temps.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return load_production_model()


def save_prediction_to_history(
    text: str,
    model_name: str,
    predicted_label: str,
    confidence: float | None,
    mode: str,
) -> None:
    st.session_state.history.insert(
        0,
        {
            "text": text,
            "model": model_name,
            "label": predicted_label,
            "confidence": confidence,
            "mode": mode,
        },
    )
    st.session_state.history = st.session_state.history[:MAX_HISTORY]


def render_probability_table(prob_df: pd.DataFrame) -> None:
    display_df = prob_df.copy()
    display_df["Probability"] = (display_df["Probability"] * 100).round(2).astype(str) + "%"
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_sample_text_buttons() -> str:
    st.markdown(
        """
        <div class="section-box">
            <div class="section-title">Exemples de textes rapides</div>
            <div class="section-text">
                Clique sur l'un des exemples ci-dessous pour tester rapidement l'interface.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _SAMPLES = {
        "Charger un exemple type Schizophrénie": (
            "I feel like people are watching me all the time, "
            "and sometimes I hear voices even when no one is around."
        ),
        "Charger un exemple type Dépression": (
            "I have been feeling hopeless, exhausted, and empty for weeks, "
            "and I struggle to get out of bed."
        ),
        "Charger un exemple type Bipolarité": (
            "My thoughts are racing, I barely sleep, and I feel like I can do anything right now."
        ),
    }

    selected_sample = ""
    cols = st.columns(len(_SAMPLES))
    for col, (label, text) in zip(cols, _SAMPLES.items()):
        with col:
            if st.button(label, use_container_width=True):
                selected_sample = text

    return selected_sample


def answer_with_openrouter(user_prompt: str) -> str:
    system_prompt = (
        "You are a careful assistant for a mental health NLP project dashboard. "
        "You help explain the project, metrics, model choices, deployment logic, and ethical framing. "
        "Do not present outputs as medical diagnosis. Be concise, clear, and professional. "
        "Always answer in French, regardless of the language of the question."
    )
    return ask_llm(prompt=user_prompt, system_prompt=system_prompt)


# ============================================================
# Barre latérale
# ============================================================

def build_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.4rem 0 1rem 0;">
                <div style="font-size: 1.35rem; font-weight: 800; color: #f8fafc;">Mental Health</div>
                <div style="color: #94a3b8; font-size: 0.92rem;">Tableau de bord clinique NLP</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected = option_menu(
            menu_title=None,
            options=["Vue d'ensemble", "Prédictions", "Suivi", "Discussion", "Historique", "À propos"],
            icons=[
                "grid-fill", "activity", "bar-chart-line-fill",
                "chat-dots-fill", "clock-history", "info-circle-fill",
            ],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#67e8f9", "font-size": "18px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "6px 0px",
                    "padding": "12px 14px",
                    "border-radius": "12px",
                    "color": "#cbd5e1",
                    "--hover-color": "rgba(255,255,255,0.06)",
                },
                "nav-link-selected": {
                    "background": "linear-gradient(90deg, rgba(34,211,238,0.18), rgba(59,130,246,0.18))",
                    "color": "#ffffff",
                    "font-weight": "700",
                },
            },
        )

        st.markdown("---")
        st.markdown("**État du système**")
        st.caption("Interface du tableau de bord prête")
        st.caption("Pipeline de prédiction prêt")
        st.caption("Mode démo de secours activé")
        st.caption(f"Modèle OpenRouter : {get_default_model()}")

        if st.button("Effacer l'historique des prédictions", use_container_width=True):
            st.session_state.history = []
            st.success("Historique effacé.")

    return selected


# ============================================================
# Pages
# ============================================================

def render_overview() -> None:
    render_header()
    dataset_info = load_dataset_info()
    artifacts = load_monitoring_artifacts()

    total_texts = dataset_info["total_texts"] if dataset_info["total_texts"] is not None else "N/D"
    classes = dataset_info["num_classes"]

    final_test_df = artifacts["final_test_df"]
    critical_recall = safe_get_first_value(final_test_df, ["critical_recall"])
    champion_model = safe_get_first_value(final_test_df, ["champion_model", "model_name", "model"])

    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">Dépistage précoce des risques en santé mentale à partir de texte</div>
            <div class="hero-text">
                Un tableau de bord premium pour la classification NLP en santé mentale, conçu pour présenter
                les prédictions du modèle, la préparation au déploiement, la logique de suivi et des flux de
                revue inspirés du monde clinique.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Textes totaux", str(total_texts), "État du jeu de données", "accent-cyan"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Classes", str(classes), "NLP multi-classes", "accent-purple"), unsafe_allow_html=True)
    with c3:
        st.markdown(
            metric_card("Rappel critique", format_metric(critical_recall), "Chargé depuis les métriques de test final", "accent-green"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            metric_card(
                "Modèle champion",
                str(champion_model) if champion_model is not None else "N/D",
                "Chargé depuis les artefacts",
                "accent-gold",
            ),
            unsafe_allow_html=True,
        )

    class_badges = "".join(
        [f'<span class="small-badge">{escape(label)}</span>' for label in dataset_info["class_names"]]
    )
    st.markdown(
        f"""
        <div class="section-box">
            <div class="section-title">Catégories cliniques détectées</div>
            <div class="section-text">{class_badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_message = (
        f"Connecté au jeu de données : {dataset_info['dataset_path']}"
        if dataset_info["dataset_loaded"]
        else "Fichier de données introuvable. L'application fonctionne quand même en mode présentation."
    )
    st.markdown(
        f"""
        <div class="section-box">
            <div class="section-title">Aperçu du projet</div>
            <div class="section-text">
                {escape(dataset_message)}<br><br>
                Ce tableau de bord prend en charge les flux de prédiction, le chargement de modèles,
                l'inspection des probabilités, les panneaux de suivi, l'historique de session et une
                démonstration de déploiement.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_predictions() -> None:
    render_header()

    selected_sample = render_sample_text_buttons()
    default_text = selected_sample if selected_sample else ""

    left_col, right_col = st.columns([1.4, 1])

    with left_col:
        st.markdown(
            """
            <div class="section-box">
                <div class="section-title">Classification de texte</div>
                <div class="section-text">
                    Colle un exemple de texte, choisis un modèle, et lance une prédiction.
                    « Modèle en production (MLflow) » va chercher le modèle actuellement déployé
                    (le même que sert l'API) directement dans le MLflow Model Registry. Les autres
                    options chargent un fichier de modèle classique local. Si rien n'est disponible,
                    l'application bascule automatiquement en mode démo.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        user_text = st.text_area(
            "Saisis le texte à analyser",
            value=default_text,
            height=220,
            placeholder="Écris ou colle un texte ici...",
        )

        selected_model = st.selectbox(
            "Sélectionner un modèle",
            [MLFLOW_PRODUCTION_OPTION] + list(MODEL_CANDIDATES.keys()),
            index=0,
        )

        analyse_clicked = st.button("Analyser le texte", type="primary", use_container_width=True)

    with right_col:
        st.markdown(
            """
            <div class="section-box">
                <div class="section-title">Conseils pour la prédiction</div>
                <div class="section-text">
                    Utilise des exemples en langage naturel, semblables à des publications d'utilisateurs
                    ou des récits de type patient. Les modèles classiques sont chargés en premier. Les
                    emplacements pour les transformeurs sont prêts pour une future intégration.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyse_clicked:
        if not user_text.strip():
            st.warning("Merci de saisir un texte avant de lancer l'analyse.")
            return

        with st.spinner("Prédiction en cours..."):
            if selected_model == MLFLOW_PRODUCTION_OPTION:
                # Chemin MLflow : va chercher le modèle réellement en
                # production (le même que sert l'API), au lieu d'un fichier
                # .joblib local -- voir load_mlflow_production_model().
                loaded = load_mlflow_production_model()
                model_path = (
                    f"MLflow Model Registry — mental_health_classifier v{loaded.version} (flavor={loaded.flavor})"
                    if loaded.is_available else None
                )

                if loaded.is_available:
                    try:
                        response = (
                            _predict_with_transformers_model(loaded.model, user_text)
                            if loaded.flavor == "transformers"
                            else _predict_with_sklearn_model(loaded.model, user_text)
                        )
                        predicted_label = response.label
                        confidence = response.confidence
                        if response.probabilities:
                            prob_df = (
                                pd.DataFrame(
                                    {"Class": list(response.probabilities.keys()),
                                     "Probability": list(response.probabilities.values())}
                                )
                                .sort_values("Probability", ascending=False)
                                .reset_index(drop=True)
                            )
                        else:
                            prob_df = pd.DataFrame({"Class": [predicted_label], "Probability": [confidence]})
                        mode = "real"
                        load_error = None
                    except Exception as exc:
                        st.error(f"La prédiction a échoué avec le modèle de production MLflow : {exc}")
                        predicted_label, confidence, prob_df = fallback_demo_prediction(user_text)
                        mode = "demo-fallback"
                        load_error = None
                else:
                    predicted_label, confidence, prob_df = fallback_demo_prediction(user_text)
                    mode = "demo"
                    load_error = f"Aucun modèle 'production' disponible dans MLflow : {loaded.error}"
            else:
                # Chemin classique : fichier .joblib local (voir services.py).
                model, model_path, load_error = load_joblib_model(selected_model)

                if model is not None:
                    try:
                        predicted_label, confidence, prob_df = predict_with_model(model, user_text)
                        mode = "real"
                    except Exception as exc:
                        st.error(f"La prédiction a échoué avec le modèle chargé : {exc}")
                        predicted_label, confidence, prob_df = fallback_demo_prediction(user_text)
                        mode = "demo-fallback"
                else:
                    predicted_label, confidence, prob_df = fallback_demo_prediction(user_text)
                    mode = "demo"

        save_prediction_to_history(
            text=user_text,
            model_name=selected_model,
            predicted_label=predicted_label,
            confidence=confidence,
            mode=mode,
        )

        st.session_state.last_prediction = {
            "text": user_text,
            "model": selected_model,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "mode": mode,
            "model_path": str(model_path) if model_path else None,
        }

        result_col, details_col = st.columns([1, 1])

        with result_col:
            confidence_text = f"{confidence * 100:.2f}%" if confidence is not None else "N/D"
            mode_label = "Modèle réel" if mode == "real" else "Mode démo"

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">Résultat de la prédiction</div>
                    <div class="result-value">{escape(predicted_label)}</div>
                    <div class="result-label">Confiance</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #22d3ee; margin-bottom: 0.9rem;">{escape(confidence_text)}</div>
                    <div class="divider-line"></div>
                    <div class="result-label">Modèle sélectionné</div>
                    <div style="color: #f8fafc; font-weight: 700; margin-bottom: 0.6rem;">{escape(selected_model)}</div>
                    <div class="result-label">Mode d'exécution</div>
                    <div style="color: #cbd5e1; font-weight: 600;">{escape(mode_label)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if load_error:
                st.info(load_error)
            if model_path:
                st.caption(f"Chargé depuis : {model_path}")

        with details_col:
            st.markdown(
                """
                <div class="section-box">
                    <div class="section-title">Probabilités par classe</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_probability_table(prob_df)

        st.markdown(
            """
            <div class="section-box">
                <div class="section-title">Texte soumis</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(user_text)


def render_monitoring() -> None:
    render_header()
    artifacts = load_monitoring_artifacts()

    using_sample_outputs = any(
        not path.exists() and (SAMPLE_OUTPUTS_DIR / path.name).exists()
        for path in [
            FINAL_TEST_METRICS_PATH,
            NESTED_CV_SUMMARY_PATH,
            NORMAL_CV_SUMMARY_PATH,
            GLOBAL_CLINICAL_REVIEW_PATH,
        ]
    )

    if using_sample_outputs:
        st.info(
            "Affichage d'exemples de résultats d'évaluation depuis docs/sample_outputs/. "
            "Exécute les notebooks pour générer les artefacts locaux complets dans reports/tables/."
        )

    final_test_df    = artifacts["final_test_df"]
    nested_cv_df     = artifacts["nested_cv_df"]
    normal_cv_df     = artifacts["normal_cv_df"]
    global_review_df = artifacts["global_review_df"]

    macro_f1       = safe_get_first_value(final_test_df, ["f1_macro", "macro_f1"])
    macro_recall   = safe_get_first_value(final_test_df, ["recall_macro", "macro_recall"])
    critical_recall = safe_get_first_value(final_test_df, ["critical_recall"])
    champion_model = safe_get_first_value(final_test_df, ["champion_model", "model_name", "model"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Macro F1", format_metric(macro_f1), "Métrique de test final", "accent-cyan"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Rappel macro", format_metric(macro_recall), "Métrique de test final", "accent-green"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Rappel critique", format_metric(critical_recall), "Classes prioritaires", "accent-gold"), unsafe_allow_html=True)
    with c4:
        st.markdown(
            metric_card(
                "Modèle champion",
                str(champion_model) if champion_model is not None else "N/D",
                "Chargé depuis les artefacts",
                "accent-purple",
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="section-box">
            <div class="section-title">Panneau de suivi</div>
            <div class="section-text">
                Affiche les artefacts d'évaluation réels lorsque les fichiers CSV sont présents dans les dossiers du projet.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Métriques de test final")
    if final_test_df is not None:
        st.dataframe(final_test_df, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Fichier manquant : {FINAL_TEST_METRICS_PATH}")

    st.markdown("### Résumé de la CV imbriquée")
    if nested_cv_df is not None:
        st.dataframe(nested_cv_df, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Fichier manquant : {NESTED_CV_SUMMARY_PATH}")

    st.markdown("### Résumé de la CV normale")
    if normal_cv_df is not None:
        st.dataframe(normal_cv_df, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Fichier manquant : {NORMAL_CV_SUMMARY_PATH}")

    st.markdown("### Revue clinique globale")
    if global_review_df is not None:
        st.dataframe(global_review_df, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Fichier manquant : {GLOBAL_CLINICAL_REVIEW_PATH}")


def render_chat() -> None:
    render_header()

    st.markdown(
        """
        <div class="section-box">
            <div class="section-title">Copilote du projet</div>
            <div class="section-text">
                Pose des questions sur le projet, son objectif, la logique de triage, la valeur métier,
                les métriques, l'approche de développement ou la configuration de déploiement.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Modèle LLM configuré : {get_default_model()}")

    if st.session_state.qa_chain is None:
        with st.spinner("Chargement de la base de connaissances..."):
            try:
                st.session_state.qa_chain = build_qa_chain()
            except Exception:
                st.session_state.qa_chain = None

    if st.button("Effacer la discussion", use_container_width=False):
        st.session_state.chat_messages = []
        st.success("Discussion effacée.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Pose une question sur ton projet...")

    if user_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Réflexion en cours..."):
                answer = ""

                if st.session_state.qa_chain is not None:
                    try:
                        result = st.session_state.qa_chain.invoke({"query": user_prompt})
                        answer = result["result"]

                        source_docs = result.get("source_documents", [])
                        if source_docs:
                            names = [Path(d.metadata.get("source", "")).name for d in source_docs if d.metadata.get("source")]
                            unique_names = list(dict.fromkeys(names))
                            if unique_names:
                                answer += "\n\n**Sources :** " + ", ".join(unique_names)
                    except Exception as exc:
                        answer = f"Le pipeline RAG a échoué : {exc}\n\nBascule vers OpenRouter...\n\n"

                if not answer:
                    try:
                        answer = answer_with_openrouter(user_prompt)
                    except Exception as exc:
                        answer = (
                            "Impossible de répondre, ni avec l'assistant RAG ni avec OpenRouter.\n\n"
                            f"Erreur OpenRouter : {exc}\n\n"
                            "Vérifie ton fichier .env et OPENROUTER_API_KEY."
                        )

                st.markdown(answer)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})


def render_history() -> None:
    render_header()

    st.markdown(
        """
        <div class="section-box">
            <div class="section-title">Historique des prédictions</div>
            <div class="section-text">
                Stocke uniquement les prédictions les plus récentes de la session en cours.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.history:
        st.info("Aucune prédiction pour l'instant dans cette session.")
        return

    for item in st.session_state.history:
        confidence_text = (
            f"{item['confidence'] * 100:.2f}%"
            if item["confidence"] is not None
            else "N/D"
        )
        st.markdown(
            f"""
            <div class="history-card">
                <div class="history-title">{escape(str(item['label']))}</div>
                <div class="history-meta">
                    Modèle : {escape(str(item['model']))} | Confiance : {escape(confidence_text)} | Mode : {escape(str(item['mode']))}
                </div>
                <div class="history-text">{escape(str(item['text']))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_about() -> None:
    render_header()

    st.markdown(
        """
        <div class="section-box">
            <div class="section-title">À propos de ce tableau de bord</div>
            <div class="section-text">
                Cette application Streamlit est une interface de démonstration professionnelle pour un projet
                de classification NLP en santé mentale. Elle combine un design visuel soigné avec une logique
                de déploiement concrète : chargement de modèle, prédiction de texte, affichage de la confiance,
                artefacts de suivi et historique de session.
                <br><br>
                Elle est destinée à des fins de démonstration, de portfolio et de présentation MVP.
                <strong>Ce n'est pas un dispositif de diagnostic.</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-box">
            <div class="section-title">Prochaines étapes recommandées</div>
            <div class="section-text">
                1. Connecter ton vrai modèle classique sauvegardé (.joblib dans models/).<br>
                2. Ajouter des points de terminaison d'inférence transformeurs ou des pipelines transformeurs.<br>
                3. Afficher les top-k classes et des panneaux d'explication plus riches.<br>
                4. Ajouter des graphiques à partir des vrais fichiers CSV d'évaluation.<br>
                5. Déployer sur Streamlit Community Cloud ou une autre plateforme.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Point d'entrée principal
# ============================================================

def main() -> None:
    load_custom_css()
    page = build_sidebar()

    dispatch = {
        "Vue d'ensemble": render_overview,
        "Prédictions":    render_predictions,
        "Suivi":          render_monitoring,
        "Discussion":     render_chat,
        "Historique":     render_history,
        "À propos":       render_about,
    }

    renderer = dispatch.get(page)
    if renderer:
        renderer()


if __name__ == "__main__":
    main()
