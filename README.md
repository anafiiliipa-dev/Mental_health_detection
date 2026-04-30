<div align="center">

# 🧠 Mental Health Intelligence

### NLP mental health triage powered by ML + RAG + LLMs

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)]
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)]
[![scikit-learn](https://img.shields.io/badge/ML-ScikitLearn-F7931E?logo=scikit-learn&logoColor=white)]
[![CI](https://github.com/anafiiliipa-dev/Mental_health_detection/actions/workflows/ci.yml/badge.svg)]
[![Tests](https://img.shields.io/badge/tests-33%20passed-brightgreen)]

</div>

---

## 🚀 What is this?

An **AI-powered decision-support tool** that classifies mental health-related text into 7 clinical categories:

> ADHD · Anxiety · Autism · Bipolar · BPD · Depression · Schizophrenia

Built with:

- ✔️ Classical ML (LinearSVC + TF-IDF)
- ✔️ Nested Cross-Validation (no bias)
- ✔️ RAG (FAISS + LangChain)
- ✔️ LLM integration (OpenRouter)
- ✔️ Streamlit dashboard

---

## 🧪 Demo Mode

If no trained model is available, the system uses:

```python
fallback_demo_prediction()
