<div align="center">

# 🧠 Mental Health Intelligence

### Triage NLP de niveau clinique avec du ML statistiquement robuste et des LLM ancrés

[![CI](https://github.com/anafiiliipa-dev/Mental_health_detection/actions/workflows/ci.yml/badge.svg)](https://github.com/anafiiliipa-dev/Mental_health_detection/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking%20%26%20registry-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-DistilBERT-FFD21E)](https://huggingface.co/docs/transformers)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-0467DF)](https://github.com/facebookresearch/faiss)
[![Docker](https://img.shields.io/badge/Docker-Cloud%20Run-2496ED?logo=docker&logoColor=white)](https://cloud.google.com/run)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Un outil d'aide à la décision pour le dépistage précoce, construit autour d'une base de ML classique validée par Nested-CV et d'un candidat DistilBERT fine-tuné, servis via un registre MLflow gouverné et un déploiement FastAPI + GCP Cloud Run — avec un copilote LLM en Retrieval-Augmented Generation intégré dans un tableau de bord Streamlit au style glassmorphism.**

[Aperçu](#-aperçu) · [Problème](#-le-problème) · [Architecture](#-architecture) · [Méthodologie](#-méthodologie) · [Résultats](#-résultats) · [Mise en service &amp; MLOps](#-mise-en-service--mlops) · [Tableau de bord](#-tableau-de-bord) · [Démarrage rapide](#-démarrage-rapide) · [Docker](#-lancer-avec-docker) · [Éthique](#-éthique--limites)

</div>

> [!CAUTION]
> **Avertissement non diagnostique.** Ce système est une aide à la décision clinique. Il ne doit **jamais** remplacer le jugement d'un clinicien agréé, et n'est certifié comme dispositif médical sous aucun cadre réglementaire (EU MDR, US FDA, ou équivalent). Voir [`NOTICE.md`](NOTICE.md) pour l'avis complet.

---

## 🚀 Aperçu

**Mental Health Intelligence** est un système NLP de bout en bout conçu pour **trier des textes liés à la santé mentale** — récits de patients, publications sur les réseaux sociaux, formulaires d'admission — selon une approche en couches :

- 🧮 **Base de ML classique** — TF-IDF + LinearSVC, validée par Nested Cross-Validation
- 🤖 **Candidats transformers** — BERT / MentalBERT (benchmark) et un DistilBERT fine-tuné (candidat enregistré dans le pipeline MLOps), évalués face à face avec la base classique
- ⚙️ **Pipeline MLOps gouverné** — suivi et registre MLflow, un chemin de promotion contrôlé, et une mise en service FastAPI adaptée au type de modèle sur GCP Cloud Run
- 📚 **Retrieval-Augmented Generation** — FAISS + LangChain sur une base de connaissances organisée
- 🩺 **Évaluation de style clinique** — matrices de confusion par classe, analyse d'erreurs, métriques orientées rappel

L'objectif : **allier rigueur statistique et interprétabilité concrète**, en livrant des prédictions non seulement précises, mais auditables.

---

## 🎯 Le problème

Les signaux liés à la santé mentale dans du texte libre sont notoirement difficiles à classer, car :

- Les classes sont **déséquilibrées** — certaines catégories sont 5× plus rares que d'autres
- Les différences linguistiques sont **subtiles sur le plan sémantique** et dépendantes du contexte
- Les faux négatifs sur les **classes à haut risque** (Bipolaire, Schizophrénie) ont un coût clinique asymétrique — les manquer est bien pire que d'orienter à tort un cas de Dépression vers Anxiété

Ce projet répond à ce défi avec un système pensé pour une **évaluation rigoureuse d'abord, un déploiement ensuite** : prétraitement soigné, modélisation équilibrée, validation non biaisée, et un objectif d'optimisation orienté rappel.

---

## 🧠 Classes

Le système prédit parmi **sept catégories cliniques qui se chevauchent** :

| Classe                | Priorité clinique                  | Poids de classe (équilibré) |
| -------------------- | ---------------------------------- | ----------------------- |
| 🧩 **ADHD**          | Standard                           | 0,80×                   |
| 😰 **Anxiety**       | Standard                           | 0,84×                   |
| 🧠 **Autism**        | Standard                           | 1,08×                   |
| ⚡ **Bipolar**        | **Critique** (priorité au rappel)  | 0,88×                   |
| 💔 **BPD**           | Standard                           | 1,00×                   |
| 🌧 **Depression**    | Standard                           | 1,12×                   |
| 🌀 **Schizophrenia** | **Critique** (priorité au rappel)  | **1,64×**               |

> La pondération équilibrée par classe surpondère les classes sous-représentées (notamment Schizophrenia) à l'entraînement. C'est un choix clinique délibéré, pas un artefact des données.

---

## 🏗 Architecture

Le système s'articule en deux couches qui s'appuient l'une sur l'autre : une **couche locale de recherche/tableau de bord** (Streamlit, ML classique, RAG) et une **couche MLOps** (entraînement → registre MLflow → CI/CD → Cloud Run → monitoring). Le schéma ci-dessous montre le flux MLOps de bout en bout ; le tableau des « trois chemins de raisonnement » plus bas décrit le routage interne du tableau de bord Streamlit.

<p align="center">
  <img src="docs/architecture-diagram.svg" alt="Pipeline MLOps de bout en bout : les données historiques et une Mock API alimentent un bucket S3 puis une étape de validation ; le jeu validé entraîne un modèle suivi par MLflow Tracking et versionné dans MLflow Model Registry, dont les métadonnées vivent dans Neon (PostgreSQL) ; une porte de promotion compare automatiquement le candidat au modèle en production ; GitHub Actions teste, construit et interroge périodiquement le Registry ; le déploiement tourne sur GCP Cloud Run (FastAPI servi à Streamlit et à l'utilisateur) ; les logs structurés alimentent un job Evidently planifié qui détecte une dérive et relance le cycle CI/CD." width="100%">
</p>

**Le flux, numéroté comme sur le schéma :**

1. **Sources & stockage** (01–04) — les données historiques et une Mock API (simulant un flux de production) alimentent un unique bucket S3 (préfixes distincts, avec des IAM séparés, pour les données brutes vs. les artefacts MLflow), puis une étape de validation (dédoublonnage, vérification des labels, garde-fous sur les corpus vides) produit le split train/val/test.
2. **Entraînement & MLflow** (05–09) — `train.py` entraîne le champion ; chaque run est suivi dans **MLflow Tracking** (params, métriques, artefacts) et versionné dans le **MLflow Model Registry** (`candidate → staging → production`). Les métadonnées des runs vivent dans **Neon** (PostgreSQL managé) ; les artefacts des modèles vivent dans S3. Une **porte de promotion** (`promote.py`) compare automatiquement le candidat au modèle actuellement en production sur `f1_macro` et `critical_recall` avant qu'il ne puisse passer en production — voir [Mise en service & MLOps](#-mise-en-service--mlops).
3. **CI/CD** (10) — GitHub Actions exécute `ruff check` → `pytest` → build → push vers GCP Artifact Registry à chaque push/PR, et interroge séparément le Registry pour qu'un modèle nouvellement promu soit repris sans redéploiement manuel.
4. **Déploiement & prédiction** (11–13) — GCP Cloud Run sert l'application **FastAPI** (`/health`, `/model-info`, `/predict`), que le tableau de bord **Streamlit** et les utilisateurs finaux appellent tous les deux.
5. **Monitoring & boucle de réentraînement** (14–16) — chaque appel à `/predict` écrit un log structuré (jamais le texte brut du patient — voir [Frontières de confidentialité](docs/architecture.md#8-privacy-boundaries)) ; un job Evidently planifié compare les prédictions récentes à la distribution d'entraînement et, en cas de dérive significative, relance le cycle CI/CD.

Pour le raisonnement complet derrière chacun de ces choix, voir [`docs/architecture.md`](docs/architecture.md) et [`docs/deployment.md`](docs/deployment.md) pour les étapes de déploiement Cloud Run elles-mêmes.

### Trois chemins de raisonnement indépendants (tableau de bord Streamlit)

Chaque chemin applique une isolation stricte pour protéger les données utilisateur :

| Chemin                       | Déclencheur                | Frontière de confidentialité                                                     |
| -------------------------- | ---------------------- | -------------------------------------------------------------------- |
| **Inférence ML locale**     | Page `Predictions`     | Tourne entièrement hors-ligne. Aucun texte ne quitte la machine.     |
| **RAG sur les docs du projet**  | Page `Chat` (par défaut)  | Embeddings locaux ; seuls les extraits récupérés + la question sont envoyés au LLM.  |
| **Repli LLM direct**    | Le RAG ne retourne aucun contexte | Question envoyée à OpenRouter uniquement en cas d'échec de la récupération.               |

> 📖 Pour une explication détaillée des choix de conception, des compromis, et des alternatives rejetées, voir [`docs/architecture.md`](docs/architecture.md).

---

## 🔬 Méthodologie

### Rigueur statistique : **Nested Cross-Validation**

La plupart des benchmarks ML souffrent d'un **biais de sélection** — les hyperparamètres sont ajustés sur les mêmes folds que ceux utilisés pour rapporter la performance. Nous utilisons une boucle imbriquée :

```text
Outer K-Fold (estimation du test)
  └── Inner K-Fold (GridSearch sur le fold d'entraînement uniquement)
       └── Configuration championne promue, réentraînée sur tout le fold train,
           évaluée sur le fold test mis de côté
```

Le résultat est une **estimation non biaisée de l'erreur de généralisation** — ce que le modèle ferait réellement sur du texte jamais vu.

### Modèle champion : **LinearSVC + pondération équilibrée par classe**

Un choix délibérément simple. Les benchmarks transformers (`03_transformers.ipynb`) ont montré que sur ce jeu de données, MentalBERT surpasse LinearSVC de **~3 points** en macro-F1 — mais **pour ~40× le coût d'inférence** et une interprétabilité moindre. Pour un outil de triage qui doit être auditable en contexte clinique, **LinearSVC l'emporte**.

### Objectif d'optimisation : **Critical Recall**

Nous n'optimisons délibérément pas le macro-F1 seul. Manquer un signal Bipolar ou Schizophrenia a un coût réel ; orienter à tort un cas de Dépression vers Anxiété est rattrapable en aval. La métrique composite rapportée dans `final_test_metrics.csv` pondère plus fortement le rappel sur les classes critiques (`Bipolar`, `Schizophrenia`) que sur le reste — un compromis clinique délibéré.

### Suite d'évaluation

- Exactitude, F1 macro et pondéré
- Précision, rappel, F1 par classe
- Matrices de confusion
- Analyse d'erreurs clinique (notebook `04_clinical_evaluation.ipynb`)
- Ablation de sensibilité SMOTE (notebook `02b_smote_sensitivity.ipynb`)

---

## 📊 Résultats

Tous les chiffres ci-dessous proviennent de `reports/tables/` et ont été produits par les notebooks de ce dépôt — entièrement reproductibles de bout en bout.

### Chiffres clés (jeu de test mis de côté)

| Métrique                                | LinearSVC (champion) | MentalBERT (benchmark) | BERT-base (benchmark) |
| ------------------------------------- | -------------------- | ---------------------- | --------------------- |
| Macro-F1                              | **0,779**            | 0,809                  | 0,791                 |
| Rappel macro                          | **0,779**            | 0,812                  | 0,793                 |
| **Critical Recall** (Bipolar+Schiz)   | **0,698**            | 0,756                  | 0,739                 |
| Exactitude                            | n/a                  | 0,815                  | 0,798                 |
| Latence d'inférence (ms / échantillon) | **< 5 ms**           | ~200 ms                | ~200 ms               |
| Taille du modèle                      | **~10 Mo**            | ~440 Mo                 | ~440 Mo                |

### Nested CV (moyenne sur les folds externes, variante texte brut)

| Modèle                       | Macro-F1          | Critical Recall   | Score robuste | Rang |
| --------------------------- | ----------------- | ----------------- | ------------ | ---- |
| **LinearSVC balanced** ⭐    | **0,769 ± 0,006** | **0,684 ± 0,018** | **0,734**    | **1** |
| LogReg balanced             | 0,725 ± 0,004     | 0,709 ± 0,013     | 0,720        | 2    |
| LinearSVC plain             | 0,758 ± 0,008     | 0,659 ± 0,026     | 0,717        | 3    |
| LogReg plain                | 0,754 ± 0,014     | 0,630 ± 0,034     | 0,703        | 4    |
| MultinomialNB               | 0,545 ± 0,014     | 0,350 ± 0,011     | 0,474        | 5    |

**Écarts-types inférieurs à 1 point de pourcentage** entre les folds — le modèle est exceptionnellement stable. Comparé à la base naïve Multinomial NB, le champion montre une **amélioration de +22 points de F1**, validant le pipeline TF-IDF + classifieur linéaire.

### Revue clinique (n = 2 255 prédictions)

| Indicateur                                  | Nombre |
| ------------------------------------------ | ----- |
| Total des prédictions revues                 | 2 255 |
| Total des erreurs                               | 474   |
| **Faux négatifs critiques**               | **95** |
| Faux positifs critiques                   | 63    |

L'asymétrie entre les faux négatifs (95) et les faux positifs (63) sur les classes critiques est **délibérée** — l'optimisation orientée rappel accepte davantage de fausses alertes en échange de moins de cas manqués.

### Pourquoi LinearSVC l'emporte malgré un macro-F1 légèrement inférieur

| Dimension                | LinearSVC | MentalBERT | Facteur de décision                                       |
| ------------------------ | --------- | ---------- | ----------------------------------------------------- |
| Macro-F1 (test)          | 0,779     | 0,809      | Écart de 3 points de pourcentage                            |
| Coût d'inférence           | `1×`      | `~40×`     | LinearSVC passe à l'échelle pour du triage par lot                     |
| Taille du modèle sur disque       | ~10 Mo    | ~440 Mo    | LinearSVC trivial à déployer                           |
| Interprétabilité         | Coefficients au niveau des tokens | Attention boîte noire | LinearSVC défendable devant un clinicien |
| Auditabilité des biais        | Élevée      | Faible        | LinearSVC plus facile à analyser par tranches                     |
| Sensibilité SMOTE        | Négligeable (-0,4 pt) | n/a | Les poids équilibrés par classe suffisent (notebook `02b`) |

### Phase 11 — benchmark élargi & le candidat DistilBERT

Une passe de benchmarking ultérieure et distincte (`reports/tables/classical/model_comparison.csv`) a élargi la comparaison au-delà de LinearSVC/LogReg/NB pour inclure des arbres à gradient boosté et des classifieurs sur embeddings de phrases figés, et a ajouté un **DistilBERT** fine-tuné comme candidat gouverné par le pipeline MLOps (enregistré via `train/register_distilbert.py`, évalué avec le même protocole que le reste — voir `reports/transformers/distilbert_metrics.csv`) :

| Modèle                       | Variante de texte | Macro-F1  | Rappel (macro) | Critical Recall | MCC       |
| ---------------------------- | ------------ | --------- | --------------- | ---------------- | --------- |
| Embedding_SVM                | raw          | 0,744     | 0,750            | 0,691             | 0,716     |
| LinearSVC_balanced (champion)| raw          | 0,741     | 0,738            | 0,682             | 0,714     |
| **DistilBERT_finetuned**     | raw          | **0,767** | **0,766**        | **0,689**         | **0,743** |
| Embedding_LogReg             | raw          | 0,737     | 0,746            | 0,708             | 0,708     |
| LightGBM_balanced            | raw          | 0,731     | 0,733            | 0,696             | 0,704     |
| XGBoost_balanced             | raw          | 0,713     | 0,717            | **0,720**         | 0,682     |

DistilBERT devance légèrement le champion classique en macro-F1, rappel, et MCC sur cette passe, sans que l'écart ne change à lui seul la logique de déploiement — il représente toujours ~40× le coût d'inférence et ~90× la taille de LinearSVC (voir le tableau de compromis ci-dessus). Il est évalué comme un **candidat enregistré** via la même porte de promotion MLflow que tout le reste, jamais substitué à la main — voir [Mise en service & MLOps](#-mise-en-service--mlops) pour ce que « promu » signifie concrètement ici et son statut actuel.

### Vérifications de robustesse & d'équité (modèle champion)

Deux passes d'évaluation supplémentaires éprouvent le champion au-delà du jeu de test propre (`train/robustness.py`, `train/bias_slicing.py`) :

- **Robustesse au bruit** — une injection légère de fautes de frappe coûte ~4 points de macro-F1 et ~6 points de critical recall ; des fautes lourdes coûtent respectivement ~15 et ~17 points. Les changements de casse (majuscules/minuscules/aléatoire) n'ont **aucun** effet, comme attendu pour un pipeline TF-IDF.
- **Équité par classe** — le rappel varie de 0,63 (Autism) à 0,92 (ADHD) selon les classes ; les deux classes critiques (Bipolar, Schizophrenia) se situent en milieu de tableau (0,70 et 0,66). Le rappel sur les **textes courts est nettement plus faible** (macro-F1 0,69) que sur les textes longs (macro-F1 0,79) — un écart réel à prendre en compte si l'outil est un jour utilisé sur des entrées courtes (par ex. des messages de chat) plutôt que sur les récits plus longs sur lesquels il a été entraîné.

Détails complets par classe et par longueur : `reports/tables/classical/robustness_report.csv` et `reports/tables/classical/bias_slicing_report.csv`.

---

## ⚙️ Mise en service & MLOps

Au-delà des notebooks, le champion (et tout candidat enregistré) peut être servi en direct via un pipeline gouverné plutôt qu'un simple fichier `.joblib` statique :

- **MLflow Model Registry** — chaque modèle entraîné est journalisé avec ses métriques et versionné sous `mental_health_classifier`, en passant par les alias `staging` → `production`. Rien n'est promu en cliquant à la main dans l'interface MLflow : `src/mental_health/train/promote.py` est le seul chemin vers « production », et applique une seule règle — un candidat ne peut être promu que s'il ne **régresse pas** sur `f1_macro` **et** `critical_recall` par rapport à ce qui est actuellement en service (la toute première inscription est la seule exception d'amorçage). Cela existe spécifiquement pour qu'un modèle qui sacrifierait le rappel sur Bipolar/Schizophrenia pour un F1 plus flatteur ne puisse pas devenir production discrètement.
- **Mise en service adaptée au type de modèle** — la couche FastAPI (`src/mental_health/api/`) détecte si le modèle aliasé « production » est un pipeline scikit-learn ou un pipeline 🤗 Transformers (`model_loader.py`) et adapte la prédiction en conséquence (`main.py`), de sorte qu'un modèle classique et un DistilBERT fine-tuné puissent tous deux être servis via exactement le même contrat `/predict`, sans changement de code.
- **Backend partagé pour le déploiement en équipe** — `MLFLOW_TRACKING_URI`/`MLFLOW_ARTIFACT_ROOT` pointent par défaut vers un store local SQLite/fichier (dev solo) ou vers un backend partagé **Neon Postgres + bucket compatible S3** lorsqu'ils sont définis (déploiement en équipe/Cloud Run) — même code, backend différent, sans logique conditionnelle nécessaire.
- **Confidentialité dans les logs de production** — chaque appel à `/predict` journalise une empreinte (SHA-256, tronquée) et une longueur au lieu du texte soumis, ainsi que le label prédit et la distribution complète des probabilités par classe — suffisant pour déboguer le trafic et la confiance sans jamais conserver le texte du patient côté serveur.

Endpoints une fois déployé : `GET /health` (liveness), `GET /model-info` (quelle version est en service, ses métriques, ou pourquoi aucune ne l'est), `POST /predict` (classifier un texte — se replie sur une démo heuristique clairement identifiée si aucun modèle de production n'est encore enregistré, afin que l'API ne tombe jamais en échec dur). Étapes de déploiement complètes : [`docs/deployment.md`](docs/deployment.md).

---

## 🎨 Tableau de bord

Une application Streamlit à six pages avec un thème glassmorphism personnalisé :

| Page            | Ce qu'elle fait                                                                                                                        |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Overview**    | Statistiques du jeu de données en direct, badges de classe, métriques clés du dernier run d'évaluation.                                                  |
| **Predictions** | Collez n'importe quel texte → classification en temps réel, barres de probabilité, indicateur de confiance. Se replie sur une démo par mots-clés clairement identifiée si aucun modèle `.joblib` n'est chargé. |
| **Monitoring**  | Affiche les véritables artefacts d'évaluation (résumé Nested CV, métriques finales de test, CSV de revue clinique) — aucun chiffre inventé.             |
| **Chat**        | Questions-réponses en RAG ancrées dans `rag_source/`. Cite ses sources. Se replie sur OpenRouter direct en cas d'échec de récupération.                        |
| **History**     | Journal des prédictions récentes, limité à la session. Effacé au rechargement du navigateur — aucune persistance.                                              |
| **About**       | Contexte du projet, cadrage éthique, feuille de route.                                                                                          |

### Captures d'écran

|                         |                         |
| ----------------------- | ----------------------- |
| ![Overview](docs/screenshots/01-overview.png)     | ![Predictions](docs/screenshots/02-predictions.png) |
| ![Monitoring](docs/screenshots/03-monitoring.png) | _Chat avec RAG ancré (à venir)_              |

---

## 🚀 Démarrage rapide

### Prérequis

- **Python 3.10+**
- Une clé API [OpenRouter](https://openrouter.ai/) pour la page Chat (offre gratuite disponible avec les modèles `:free`)
- ~2 Go d'espace disque pour le modèle d'embedding + l'index FAISS au premier lancement

### Installation

```bash
git clone https://github.com/anafiiliipa-dev/Mental_health_detection.git
cd Mental_health_detection

python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate

pip install -e ".[streamlit]"      # cœur + tableau de bord
# ou :
pip install -e ".[dev]"            # cœur + tests + linters
```

### Configurer les secrets

```bash
cp .env.example .env
# Éditez .env et renseignez OPENROUTER_API_KEY
```

Le dépôt **ne commite jamais** `.env` — seulement `.env.example` (valeurs de substitution). Confirmé par `.gitignore`.

Pour une configuration entièrement gratuite, utilisez `meta-llama/llama-3.3-70b-instruct:free` dans `OPENROUTER_MODEL` — aucun crédit requis.

### Modèle

Le modèle champion entraîné (**`models/best_classical_model.joblib`**, ~5 Mo) est **inclus dans ce dépôt** afin de pouvoir lancer l'inférence immédiatement après le clone. Pour réentraîner depuis zéro, exécutez les notebooks `01 → 02` de bout en bout.

### Lancer

```bash
streamlit run src/mental_health/app/app.py
# ou, après `pip install -e .` :
mental-health-dashboard
```

Rendez-vous sur `http://localhost:8501`.

### Tester

```bash
pytest -v
```

Tous les services externes (OpenRouter, FAISS, le backend MLflow lui-même dans les tests de l'API) sont mockés ou pointent vers un store local temporaire — aucun réseau, clé API, ni déploiement MLflow/Cloud réel n'est requis pour exécuter la suite de tests.

### Lancer la couche de mise en service FastAPI (optionnel — la couche MLOps)

```bash
pip install -e ".[api,mlflow]"        # ajoutez ",transformers" aussi si un candidat DistilBERT est enregistré
uvicorn mental_health.api.main:app --reload
```

Rendez-vous sur `http://localhost:8000/docs` pour la documentation interactive de l'API. Sans rien d'enregistré dans MLflow, `/predict` répond quand même — avec une démo heuristique clairement identifiée (`is_demo_fallback: true`) plutôt qu'un échec dur. Pour entraîner et enregistrer un modèle réel, voir `src/mental_health/train/train.py` et `train/promote.py` ; pour pointer l'API vers un backend d'équipe partagé plutôt que le store local SQLite/fichier, définissez `MLFLOW_TRACKING_URI` / `MLFLOW_ARTIFACT_ROOT` dans `.env` (voir `.env.example`). Déployer ceci sur GCP Cloud Run : [`docs/deployment.md`](docs/deployment.md).

---

## 🐳 Lancer avec Docker

Déploiement sans configuration pour les évaluateurs et recruteurs. Une seule commande :

```bash
docker compose up --build
```

Rendez-vous sur `http://localhost:8501`.

Pour un lancement ponctuel personnalisé :

```bash
docker build -t mental-health-intelligence .
docker run --rm -p 8501:8501 \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -v "$(pwd)/models:/app/models:ro" \
  -v "$(pwd)/reports/tables:/app/reports/tables:ro" \
  mental-health-intelligence
```

L'image tourne avec un utilisateur non-root, expose un healthcheck `/_stcore/health`, et utilise des indices de build multi-étapes pour des rebuilds rapides.

---

## 📁 Structure du projet

```text
mental-health-intelligence/
├── .github/workflows/ci.yml           # pytest + ruff à chaque PR
├── .pre-commit-config.yaml            # hooks ruff + nbstripout
├── .env.example                       # secrets de substitution
├── .gitattributes                     # force LF, marque les notebooks
├── .gitignore                         # data/, models/ (avec exception pour le modèle), reports/ (avec exceptions CSV)
├── .dockerignore
├── pyproject.toml                     # source unique de vérité pour les dépendances
├── requirements.txt                   # raccourci pratique (miroir de pyproject)
├── requirements_streamlit.txt         # raccourci pratique (extra streamlit)
├── README.md
├── LICENSE                            # MIT pure (pour que GitHub la détecte)
├── NOTICE.md                          # avis non diagnostique
├── CHANGELOG.md
├── Dockerfile
├── docker-compose.yml
│
├── docs/
│   ├── architecture.md                # décisions de conception et compromis (couche locale/tableau de bord)
│   ├── deployment.md                  # étapes de déploiement Cloud Run (couche MLOps)
│   ├── architecture-diagram.svg       # flux MLOps de bout en bout (intégré ci-dessus)
│   ├── screenshots/                   # captures du tableau de bord pour le README
│   └── sample_outputs/                # CSV publics sûrs (agrégés uniquement)
│
├── notebooks/                         # pipeline 1️⃣ → 5️⃣, exécutable de bout en bout
│   ├── 00_exploration.ipynb
│   ├── 01_data_cleaning.ipynb
│   ├── 02_classical_ml.ipynb                 # ← champion Nested CV
│   ├── 02b_smote_sensitivity.ipynb           # ← ablation du déséquilibre des classes
│   ├── 03_transformers.ipynb                 # ← Colab/GPU
│   ├── 04_clinical_evaluation.ipynb
│   └── 05_deployment_mvp.ipynb
│
├── src/mental_health/                 # package installable
│   ├── __init__.py
│   ├── app/
│   │   ├── app.py                     # point d'entrée Streamlit (CSS + toutes les pages, fichier unique)
│   │   └── openrouter_client.py
│   ├── api/                           # couche de mise en service FastAPI (Phase 6+)
│   │   ├── main.py                    # /health, /model-info, /predict
│   │   ├── model_loader.py            # charge le modèle aliasé « production », adapté au type de modèle
│   │   ├── fallback.py                # prédiction heuristique de démo quand aucun modèle n'est enregistré
│   │   ├── logging_config.py          # logging JSON structuré, jamais le texte brut
│   │   └── schemas.py
│   ├── config/
│   │   ├── paths.py                   # constantes de chemins centralisées
│   │   └── mlflow_config.py           # URI de tracking / racine des artefacts / nom du registre, surchargeable par l'environnement
│   ├── models/
│   │   └── services.py                # load_model, predict_with_model
│   ├── train/                         # entraînement, benchmark, registre MLflow, promotion (Phase 3+)
│   │   ├── train.py                   # run d'entraînement du champion, journalisé dans MLflow
│   │   ├── benchmark.py               # comparaison multi-modèles en Nested CV
│   │   ├── promote.py                 # le seul chemin vers l'alias « production »
│   │   ├── register_distilbert.py     # enregistre le DistilBERT fine-tuné comme candidat staging
│   │   ├── distilbert_finetune.py     # fine-tune distilbert-base-uncased (extra transformers)
│   │   ├── robustness.py              # évaluation par perturbation de typos/casse
│   │   └── bias_slicing.py            # tranches d'équité par classe et par longueur
│   ├── monitoring/                    # détection de dérive sur le modèle MLflow partagé (Phase 10)
│   │   ├── mock_stream.py             # simule un flux de trafic de production
│   │   └── drift_check.py             # job de dérive Evidently
│   └── rag/
│       └── simple_rag.py              # récupération FAISS + LangChain
│
├── tests/                             # pytest, entièrement mocké — voir le badge CI ci-dessus pour le nombre à jour
│   ├── test_predictions.py
│   ├── test_rag.py
│   ├── test_client.py
│   ├── test_main.py                   # tests d'intégration FastAPI
│   ├── test_model_loader.py
│   ├── test_train.py / test_promote.py / test_register_distilbert.py
│   ├── test_robustness.py / test_bias_slicing.py
│   └── ...
│
├── models/
│   ├── best_classical_model.joblib    # champion LinearSVC (commité, ~5 Mo)
│   └── distilbert_finetuned/          # checkpoint DistilBERT fine-tuné (sortie d'entraînement local)
│
├── reports/
│   ├── tables/
│   │   ├── classical/
│   │   │   ├── final_test_metrics.csv
│   │   │   ├── model_comparison.csv        # benchmark élargi Phase 11
│   │   │   ├── robustness_report.csv
│   │   │   ├── bias_slicing_report.csv
│   │   │   ├── nested_cv_summary.csv
│   │   │   └── normal_cv_summary.csv
│   │   └── clinical/
│   │       └── global_comparison_for_clinical_review.csv
│   └── transformers/
│       └── distilbert_metrics.csv          # évaluation du candidat DistilBERT
│
├── scripts/
│   └── publish_to_shared_backend.py   # ponctuel : republie le modèle de production local vers le backend MLflow partagé
│
├── rag_source/                        # base de connaissances pour la page Chat
└── data/                              # gitignored (local uniquement)
```

---

## 🧪 Pipeline de notebooks

| #    | Notebook                    | Objectif                                                        | Durée          |
| ---- | --------------------------- | -------------------------------------------------------------- | ---------------- |
| 00   | `exploration`               | Exploration initiale du jeu de données sur Colab                           | CPU/Colab        |
| 01   | `data_cleaning`             | Normalisation du texte, détection de quasi-doublons, export train/val/test | CPU, ~2 min |
| 02   | `classical_ml`              | TF-IDF + LinearSVC, **Nested CV**, sélection du champion          | CPU, ~15 min     |
| 02b  | `smote_sensitivity`         | Teste si le sur-échantillonnage améliore le rappel sur les classes critiques | CPU, ~5 min      |
| 03   | `transformers`              | BERT base + MentalBERT, comparaison équitable avec la base classique | **GPU recommandé** |
| 04   | `clinical_evaluation`       | Matrices de confusion par classe, analyse d'erreurs, tables de revue clinique | CPU, ~3 min |
| 05   | `deployment_mvp`            | Perspectives LLM, câblage du MVP                                   | CPU, ~2 min      |

> Lancez le notebook 03 sur Google Colab via le badge en haut de chaque notebook.

---

## 🛡 Éthique & limites

- **Non diagnostique.** Répété partout où c'est important. Les sorties sont des signaux pour une revue humaine, pas des étiquettes.
- **Confidentialité.** L'inférence classique est entièrement locale. La couche RAG n'envoie que *les extraits récupérés plus la question de l'utilisateur* à OpenRouter — jamais le texte brut du patient. La page `Predictions` n'appelle jamais aucune API externe.
- **Orienté rappel.** Nous acceptons davantage de faux positifs (63) en échange de moins de faux négatifs (95) sur les classes à haut risque. C'est un compromis clinique délibéré, pas un bug.
- **Provenance des données.** Les données d'entraînement proviennent de texte publiquement disponible ; aucun dossier clinique n'a été utilisé. Le modèle **n'a pas été validé sur des populations cliniques** et n'est pas certifié pour un usage clinique.
- **Biais.** Comme pour tout classifieur de texte, la performance varie selon les données démographiques, les dialectes, et les sous-populations cliniques. L'analyse d'erreurs du notebook 04 met en évidence ces écarts ; ne pas déployer sans revalider sur votre propre population.

Voir [`NOTICE.md`](NOTICE.md) pour l'avis non diagnostique complet.

---

## 🗺 Feuille de route

- [x] MLflow Tracking + Model Registry, avec un chemin de promotion explicite et contrôlé par script (`staging` → `production`)
- [x] Sorties de probabilité calibrées (Platt scaling) + Brier score / ECE sur le benchmark classique
- [x] Benchmark élargi : arbres à gradient boosté (XGBoost, LightGBM) et classifieurs sur embeddings de phrases figés
- [x] Candidat DistilBERT fine-tuné, évalué et enregistré via le même chemin de gouvernance que les modèles classiques
- [x] Robustesse (perturbation de typos/casse) et tranches d'équité par classe/longueur sur le champion
- [x] Couche de mise en service FastAPI avec chargement de modèle adapté au type (scikit-learn ou 🤗 Transformers) et repli heuristique quand rien n'est enregistré
- [x] Logging JSON structuré et préservant la confidentialité sur chaque prédiction
- [x] CI/CD : GitHub Actions → GCP Artifact Registry → Cloud Run
- [x] Backend MLflow partagé (Neon Postgres + stockage compatible S3) pour le déploiement en équipe, en plus du store local par défaut pour le dev solo
- [ ] Job de détection de dérive Evidently câblé de bout en bout pour relancer automatiquement le CI/CD en cas de dérive détectée
- [ ] Étendre `docs/architecture.md` pour couvrir la couche MLOps (MLflow, FastAPI, Cloud Run, monitoring) en plus de la conception locale/tableau de bord existante
- [ ] Capturer une capture d'écran de la page Chat (RAG avec sources citées)
- [ ] Ajouter un export ONNX du champion LinearSVC pour une mise en service sous la milliseconde
- [ ] Matrice CI conteneurisée sur Python 3.10 / 3.11 / 3.12
- [ ] Support multilingue — triage de texte en portugais / espagnol
- [ ] Démo publique sur Hugging Face Spaces

---

## 🤝 Contribuer

Ceci est un projet portfolio, mais les issues et PR sont les bienvenues. Merci de :

1. Lancer `pytest -v` et `ruff check .` avant d'ouvrir une PR.
2. Utiliser les [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
3. Si vous touchez à un notebook, lancer `nbstripout <fichier>.ipynb` pour éviter d'alourdir les diffs avec les sorties d'exécution.
4. La CI doit passer au vert avant relecture.

---

## 👤 Auteure

**Ana Gouveia** — cohorte DSFS-OD-14
[GitHub @anafiiliipa-dev](https://github.com/anafiiliipa-dev)

---

## 📄 Licence

MIT — voir [LICENSE](LICENSE). Avis non diagnostique dans [NOTICE.md](NOTICE.md).
