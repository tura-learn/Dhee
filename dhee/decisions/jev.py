"""A client for typed decisions: Jev, from TypeSafe.

Jev is a "System One" model. It never writes prose: it takes a ``state`` and
named ``questions`` and returns one typed answer per question — a Choice with a
probability for every option, or a Noul (the probability that a statement is
true). That is the right shape for the choices a memory makes on every write
and every read, and the wrong shape for anything that has to be written.

Three rules, because this sits on paths that must never fail because of it:

* **It never raises.** Every fault — no key, a timeout, an HTTP error, a shape
  this module does not recognise — is ``None``, and callers treat ``None`` as
  "do what you did before decisions existed".
* **It never retries.** The point is an answer in under a second; a retry
  spends the budget twice and lands after the caller should have moved on.
* **It is off unless configured.** ``DecisionConfig.enabled`` is False by
  default, and ``DHEE_DECISIONS=off`` switches every decider in the process
  off without touching config.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_MODEL = "typesafe/jev-1.13"
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_MODEL = "jev-latest"

#: Questions per request. Jev evaluates questions in parallel, so a batch costs
#: about the time of one; the cap keeps a single request's state and question
#: text well inside the model's context.
MAX_QUESTIONS_PER_CALL = 48

DISABLE_ENV = "DHEE_DECISIONS"


class Decider(Protocol):
    """Anything that answers typed questions about a state, or returns None."""

    def decide(self, state: Any, questions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ...


def decisions_switched_off() -> bool:
    return str(os.environ.get(DISABLE_ENV, "")).strip().lower() in {"0", "off", "false", "no"}


class JevDecider:
    """Jev over OpenRouter's decisions route or TypeSafe's own API."""

    def __init__(
        self,
        *,
        api_key: str,
        provider: str = "openrouter",
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.provider = provider
        if provider == "typesafe":
            self.url = base_url or TYPESAFE_URL
            self.model = model or TYPESAFE_MODEL
        else:
            self.url = base_url or OPENROUTER_URL
            self.model = model or OPENROUTER_MODEL
        self.timeout_seconds = float(timeout_seconds)
        #: (calls, failures, seconds) — read by status endpoints and tests.
        self.stats = {"calls": 0, "failures": 0, "seconds": 0.0}

    def decide(self, state: Any, questions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not questions or decisions_switched_off() or not self.api_key:
            return None
        answers: Dict[str, Any] = {}
        items = list(questions.items())
        for start in range(0, len(items), MAX_QUESTIONS_PER_CALL):
            chunk = dict(items[start : start + MAX_QUESTIONS_PER_CALL])
            got = self._call(state, chunk)
            if got is None:
                return None
            answers.update(got)
        return answers

    def _call(self, state: Any, questions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        import requests

        started = time.monotonic()
        self.stats["calls"] += 1
        try:
            response = requests.post(
                self.url,
                json={"model": self.model, "state": state, "questions": questions},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout_seconds,
            )
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:160]}")
            answers = (response.json() or {}).get("answers")
            if not isinstance(answers, dict):
                raise RuntimeError("no answers in the response")
            return answers
        except Exception as exc:  # noqa: BLE001 - a decision is never worth a failure
            self.stats["failures"] += 1
            logger.info("Decision model unavailable (%s); falling back", exc)
            return None
        finally:
            self.stats["seconds"] += time.monotonic() - started


def decider_from_config(config: Any) -> Optional[JevDecider]:
    """A decider for a :class:`~dhee.configs.base.DecisionConfig`, or None."""
    if config is None or not getattr(config, "enabled", False) or decisions_switched_off():
        return None
    key = getattr(config, "api_key", None) or ""
    env = getattr(config, "api_key_env", None)
    if not key and env:
        key = os.environ.get(env, "")
    if not key:
        provider = getattr(config, "provider", "openrouter")
        key = os.environ.get("TYPESAFE_API_KEY" if provider == "typesafe" else "OPENROUTER_API_KEY", "")
    if not key.strip():
        return None
    return JevDecider(
        api_key=key,
        provider=getattr(config, "provider", "openrouter"),
        base_url=getattr(config, "base_url", None),
        model=getattr(config, "model", None),
        timeout_seconds=getattr(config, "timeout_seconds", 10.0),
    )


# ----- reading answers ------------------------------------------------------


def noul(answers: Optional[Dict[str, Any]], question_id: str) -> Optional[float]:
    """The probability a Noul question came back with, or None."""
    if not answers:
        return None
    answer = answers.get(question_id)
    if not isinstance(answer, dict):
        return None
    try:
        return float(answer.get("noul"))
    except (TypeError, ValueError):
        return None


def ranked(answers: Optional[Dict[str, Any]], question_id: str) -> List[Tuple[str, float]]:
    """A Choice answer's distribution, most likely first. Empty when absent."""
    if not answers:
        return []
    answer = answers.get(question_id)
    if not isinstance(answer, dict):
        return []
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        pairs = []
        for option, p in probabilities.items():
            try:
                pairs.append((str(option), float(p)))
            except (TypeError, ValueError):
                continue
        return sorted(pairs, key=lambda pair: pair[1], reverse=True)
    choice = answer.get("choice")
    if isinstance(choice, str):
        try:
            return [(choice, float(answer.get("confidence", 0.0)))]
        except (TypeError, ValueError):
            return [(choice, 0.0)]
    return []
