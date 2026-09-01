# syntax=docker/dockerfile:1

# Phase 7: containerizes the FastAPI service (Phase 6) only — Streamlit
# runs locally, never containerized (docs/architecture.md). Cloud Run
# deployment (Phase 13) now points MLFLOW_TRACKING_URI / MLFLOW_ARTIFACT_ROOT
# at the shared Neon Postgres + S3-compatible backend instead of local
# SQLite — those are read from the environment at container start (see
# config/mlflow_config.py), never baked into the image. Set them (and the
# AWS_* S3 credentials) as Cloud Run env vars / secrets at deploy time; see
# docs/deployment.md.
FROM python:3.12-slim

# PYTHONUNBUFFERED: log lines show up immediately, not buffered until the
# container stops — matters for `docker logs` during local debugging.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy dependency metadata first so Docker's layer cache is reused on
# rebuilds unless pyproject.toml itself changes — avoids reinstalling
# scikit-learn/mlflow/fastapi on every source-code edit.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Only the API + MLflow client are needed to *serve* a model — no dev/test
# tooling, no Streamlit, no transformers/RAG stack in this image.
RUN pip install --no-cache-dir -e ".[api,mlflow]"

EXPOSE 8000

# --host 0.0.0.0 is required in a container: the uvicorn default
# (127.0.0.1) only accepts connections from inside the container itself,
# which would make the API unreachable from the host.
CMD ["uvicorn", "mental_health.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
