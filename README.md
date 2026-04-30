<div align="center">

# 🧠 Mental Health Intelligence

### Clinical-grade NLP triage with statistically robust ML and grounded LLMs

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-0467DF)](https://github.com/facebookresearch/faiss)
[![Tests](https://img.shields.io/badge/tests-pytest-33%20passed-brightgreen?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![CI](https://github.com/anafiiliipa-dev/Mental_health_detection/actions/workflows/ci.yml/badge.svg)]
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**An early-screening decision-support tool that combines a Nested-CV-validated classical ML baseline with a Retrieval-Augmented LLM copilot — wrapped in a glassmorphism Streamlit dashboard.**

[Quickstart](#-quickstart) · [Architecture](#-architecture) · [Methodology](#-methodology) · [Dashboard](#-dashboard) · [Ethics](#-ethics--limitations)

</div>

> [!WARNING]
> **Non-diagnostic disclaimer.** This system is a clinical decision-support aid. It must **never** replace a licensed clinician's judgement, and is not certified as a medical device under the EU MDR or equivalent frameworks.

---

## 🧪 Demo Mode

If no trained model is available locally, the application falls back to a heuristic-based predictor:

```python
fallback_demo_prediction()
