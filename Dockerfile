# syntax=docker/dockerfile:1

# Phase 7: containerizes the FastAPI service (Phase 6) only. MLflow stays
# local SQLite — mlflow.db / mlruns/ are mounted as a volume at `docker run`
# time (see docker-compose.yml), never baked into the image. A shared
# Postgres + S3 backend is deferred to Phase 13 (AWS), per the project's
# explicit "AWS not before Phase 13" rule — bringing it forward here would
# be scope creep this phase doesn't need.
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
