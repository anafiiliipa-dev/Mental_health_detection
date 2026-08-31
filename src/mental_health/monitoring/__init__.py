"""
Monitoring package (Phase 10 completion): the drift-detection loop.

- ``mock_stream``: simulates the arrival of new incoming messages by
  sampling from the training dataset's held-out test split, since the
  project has no live traffic yet (documented stand-in — architecture
  diagram nodes 02/15/16).
- ``drift_check``: scores a sampled batch with the current "production"
  model and compares it against the training reference set using
  Evidently, producing an HTML report and an MLflow-logged summary.

Deliberately does NOT trigger retraining — that decision belongs to
Phase 12 (automated retraining trigger), not here.
"""
