# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- Repository-level cleanup and refactor pass.

---

## [0.3.0] — 2026-05-01

### Changed

- **BREAKING:** moved source layout from `src/{app,config,models,rag}/` to a single namespaced package `src/mental_health/`. All imports now use `from mental_health.<subpackage> import ...` instead of `from src.<subpackage> import ...`. This aligns the codebase with the package name declared in `pyproject.toml` and fixes the previously broken `setuptools.packages.find` configuration. Run `pip install -e .` to pick up the new structure.
- Reorganised `LICENSE` to be pure MIT so GitHub's licence detector classifies it correctly. The non-diagnostic notice has been moved to a dedicated `NOTICE.md`.
- Reduced authorship to sole contributor (Ana Gouveia).

### Added

- `Dockerfile` and `docker-compose.yml` for one-command deployment.
- `.dockerignore` for lean image builds.
- `.pre-commit-config.yaml` wiring `ruff` (lint + format) and `nbstripout`.
- `docs/architecture.md` with detailed design decisions, trade-offs, and rejected alternatives.
- `CHANGELOG.md` (this file).
- README sections: `## 📊 Results` (with placeholder metrics for actual numbers), `## 🐳 Run with Docker`, `## 🗺 Roadmap`.
- CI badge at the top of the README.
- `coverage` configuration in `pyproject.toml`.

### Fixed

- README project-structure section now reflects the actual codebase layout.
- Removed misleading "26 tests pass" hardcoded count from the README; now relies on the live CI badge.
- Removed the `sys.path.insert(...)` hack from test modules — tests now rely on the editable install.

---

## [0.2.0] — Earlier

### Added

- Streamlit dashboard with six pages (Overview, Predictions, Monitoring, Chat, History, About) with custom glassmorphism theme.
- LinearSVC champion model with class-balanced weighting, validated via Nested Cross-Validation.
- SMOTE sensitivity ablation (`02b_smote_sensitivity.ipynb`).
- Transformer benchmark (`03_transformers_benchmark.ipynb`) with BERT base and MentalBERT.
- Clinical evaluation notebook (`04_clinical_evaluation.ipynb`) with per-class confusion matrices and error analysis.
- RAG copilot — FAISS vector store over `rag_source/`, MiniLM-L6-v2 embeddings, OpenRouter `gpt-4o-mini` LLM with retrieval-miss fallback.
- 26 mocked unit tests covering predictions, RAG retrieval, and the OpenRouter client.
- GitHub Actions CI workflow running `pytest` and `ruff` on every push.
- `.env.example` with placeholder secrets; `.gitignore` excluding all sensitive files.
- `.gitattributes` enforcing LF line endings and marking binary files.

[Unreleased]: https://github.com/anafiiliipa-dev/Mental_health_detection/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/anafiiliipa-dev/Mental_health_detection/releases/tag/v0.3.0
