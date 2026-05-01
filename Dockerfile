# =============================================================================
# Mental Health Intelligence — Streamlit Dashboard
# -----------------------------------------------------------------------------
# Production-flavoured image:
#   - non-root user
#   - healthcheck on Streamlit's internal /_stcore/health endpoint
#   - layer caching tuned: deps copied before source code
#   - slim base for fast pulls in CI / cloud
#
# Build:
#   docker build -t mental-health-intelligence .
#
# Run:
#   docker run --rm -p 8501:8501 \
#       -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
#       mental-health-intelligence
# =============================================================================

FROM python:3.11-slim AS base

# ---- OCI metadata ----------------------------------------------------------
LABEL org.opencontainers.image.title="Mental Health Intelligence"
LABEL org.opencontainers.image.description="Clinical-grade NLP triage with Nested-CV-validated ML and grounded LLMs"
LABEL org.opencontainers.image.source="https://github.com/anafiiliipa-dev/Mental_health_detection"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.authors="Ana Gouveia"

# ---- Runtime environment ---------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

WORKDIR /app

# ---- System dependencies ---------------------------------------------------
# build-essential needed for some scientific Python wheels at install time;
# curl needed for the healthcheck below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python dependencies first (better layer caching) --------------
# Copying only the metadata files maximises cache hits on iterative builds.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --upgrade pip \
    && pip install ".[streamlit]"

# ---- Copy runtime assets ---------------------------------------------------
COPY rag_source/ ./rag_source/

# ---- Drop privileges -------------------------------------------------------
RUN useradd --create-home --shell /bin/bash --uid 1001 app \
    && chown -R app:app /app
USER app

# ---- Networking & lifecycle ------------------------------------------------
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

# ---- Default command -------------------------------------------------------
CMD ["streamlit", "run", "src/mental_health/app/app.py"]
