"""Configuration de logging structuré (JSON) — Phase 10 (Monitoring).

Les logs en texte brut sont pratiques à lire dans un terminal de dev, mais difficiles à grep ou
à agréger dès que cette API tourne ailleurs (logs Docker, et éventuellement un
agrégateur de logs en Phase 13). Ceci configure le root logger pour émettre un
objet JSON par ligne à la place, avec un petit ensemble de champs fixes plus
tout ce qu'un `extra=` ajouté par un site d'appel contient.

Ceci ne change PAS ce qui est loggé, seulement la façon dont c'est formaté — les
sites d'appel (main.py) restent seuls responsables de ne jamais passer de
texte brut de requête dans `extra`. Voir `predict()` dans main.py pour l'unique
site d'appel qui compte pour la confidentialité.
"""
from __future__ import annotations

import json
import logging
import sys

# Tous les attributs qu'un LogRecord standard possède, afin de pouvoir distinguer les
# champs "extra" (ajoutés par un site d'appel via `logger.info(..., extra={...})`) des
# attributs natifs propres au record.
_RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()) | {"message", "asctime"}


class JSONFormatter(logging.Formatter):
    """Rend chaque LogRecord sous forme d'un objet JSON par ligne."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        extras = {key: value for key, value in vars(record).items() if key not in _RESERVED}
        payload.update(extras)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Fait pointer le root logger vers un unique handler stdout formaté en JSON.

    Peut être appelée plusieurs fois sans risque (par exemple une fois à l'import, puis
    de nouveau si un test a besoin de la réinitialiser) — elle remplace toujours la liste
    de handlers plutôt que d'y ajouter, afin que les logs ne soient jamais dupliqués.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
