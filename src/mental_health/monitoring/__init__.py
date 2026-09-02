"""
Package de monitoring (fin de la Phase 10) : la boucle de détection de drift.

- ``mock_stream`` : simule l'arrivée de nouveaux messages entrants en
  échantillonnant le split de test mis de côté du jeu de données d'entraînement,
  car le projet n'a pas encore de trafic réel (substitut documenté — nœuds 02/15/16
  du diagramme d'architecture).
- ``drift_check`` : score un lot échantillonné avec le modèle "production"
  actuel et le compare au jeu de référence d'entraînement en utilisant
  Evidently, produisant un rapport HTML et un résumé consigné dans MLflow.

Ne déclenche volontairement PAS de réentraînement — cette décision relève de
la Phase 12 (déclenchement automatique du réentraînement), pas d'ici.
"""
