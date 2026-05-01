# NOTICE — Mental Health Intelligence

## Non-diagnostic notice

This software is a **clinical decision-support aid**. It must **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment.

It is **not certified as a medical device** under:

- the European Union Medical Devices Regulation (EU MDR 2017/745),
- the United States Food and Drug Administration (FDA),
- the United Kingdom Medicines and Healthcare products Regulatory Agency (MHRA),
- nor any equivalent regulatory framework worldwide.

Outputs of this software are intended **only as signals for human review** and are not, under any interpretation, medical, psychological, or clinical conclusions about any individual.

## Intended use

- Educational and research purposes
- Decision-support for trained reviewers screening large volumes of free-text input
- Demonstration of NLP triage methodology

## Out of scope

- Direct-to-consumer mental health diagnosis
- Clinical decision-making without licensed professional oversight
- Any use in a setting where a misclassification could harm an individual without compensating controls

## Liability

The authors and copyright holders provide this software "AS IS", as detailed in [LICENSE](LICENSE). Users assume **all responsibility** for the consequences of using outputs of this software, including but not limited to misuse in clinical settings.

## Data provenance

- The model is trained on publicly available text data. **No clinical records were used.**
- The model **has not been validated on clinical populations**.
- Performance characteristics are estimated on the held-out test split of the same source distribution and **do not generalise** to clinical populations without re-validation.

## Bias

As with any text classifier, performance varies across demographics, dialects, age groups, and clinical sub-populations. The error analysis in `notebooks/04_clinical_evaluation.ipynb` surfaces some of these gaps but is not exhaustive.

**Do not deploy this system without re-validating on your target population.**

## Contact

If you believe this software has been used in a way that contravenes this notice, please open an issue at:
<https://github.com/anafiiliipa-dev/Mental_health_detection/issues>

---

*Last updated: 2026-05-01*
