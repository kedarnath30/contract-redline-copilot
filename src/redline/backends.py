"""Assessment backends behind one interface.

* ``OfflineBackend`` (default): deterministic playbook checks. No network, no keys.
* ``LLMBackend``: an OpenAI-compatible chat endpoint (OpenAI, Azure OpenAI, or any gateway that
  speaks the same API, e.g. a Bedrock proxy). The model's answer is validated before use: the
  risk label must be known, the quoted evidence must appear verbatim in the clause, and any
  replacement must contain the playbook's approved language. Anything that fails validation is
  escalated to a human rather than trusted.
"""

from __future__ import annotations

import json
from typing import Protocol

import httpx

from .assess import NEEDS_HUMAN, NON_COMPLIANT, RISK_LABELS, Assessment, assess, governance_flags
from .config import Settings
from .playbook import Rule


class Backend(Protocol):
    name: str

    def review(self, clause: str, rule: Rule) -> tuple[Assessment, str | None]:
        """Return the assessment and an optional proposed text (None = use playbook text)."""
        ...


class OfflineBackend:
    name = "offline"

    def review(self, clause: str, rule: Rule) -> tuple[Assessment, str | None]:
        return assess(clause, rule), None


SYSTEM_PROMPT = (
    "You review one contract clause against one rule from a company playbook. "
    "Answer only with JSON: "
    '{"risk": "compliant|fallback|non_compliant|needs_human", '
    '"reason": "<one sentence>", '
    '"evidence_quote": "<exact substring of the clause that supports the risk label>", '
    '"proposed_text": "<replacement clause if non_compliant, else empty>"}. '
    "If non_compliant, the proposed_text must include the approved language verbatim. "
    "If you are unsure, answer needs_human."
)


def build_messages(clause: str, rule: Rule) -> list[dict]:
    rule_block = json.dumps(rule.to_dict(), indent=2)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"PLAYBOOK RULE:\n{rule_block}\n\nCLAUSE:\n{clause}"},
    ]


class LLMBackend:
    name = "llm"

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        if not settings.llm_api_key:
            raise ValueError("REDLINE_LLM_API_KEY is required for the llm backend")
        self.settings = settings
        self.client = client or httpx.Client(timeout=60)

    def _request(self, messages: list[dict]) -> str:
        s = self.settings
        body = {"messages": messages, "temperature": 0, "response_format": {"type": "json_object"}}
        if s.llm_provider == "azure":
            url = (
                f"{s.llm_base_url.rstrip('/')}/openai/deployments/{s.llm_model}"
                f"/chat/completions?api-version={s.azure_api_version}"
            )
            headers = {"api-key": s.llm_api_key}
        else:
            url = f"{s.llm_base_url.rstrip('/')}/chat/completions"
            headers = {"Authorization": f"Bearer {s.llm_api_key}"}
            body["model"] = s.llm_model
        resp = self.client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def review(self, clause: str, rule: Rule) -> tuple[Assessment, str | None]:
        flags = governance_flags(clause)
        if flags:  # deterministic governance runs before any model call
            return assess(clause, rule), None
        try:
            data = json.loads(self._request(build_messages(clause, rule)))
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as exc:
            return Assessment(NEEDS_HUMAN, rule.id, None, f"Escalated: LLM error ({exc})."), None
        return validate_llm_answer(clause, rule, data)


def validate_llm_answer(clause: str, rule: Rule, data: dict) -> tuple[Assessment, str | None]:
    risk = data.get("risk")
    quote = (data.get("evidence_quote") or "").strip()
    reason = (data.get("reason") or "").strip() or "No reason given."
    if risk not in RISK_LABELS:
        return Assessment(NEEDS_HUMAN, rule.id, None, "Escalated: invalid risk label."), None
    if risk == NEEDS_HUMAN:
        return Assessment(NEEDS_HUMAN, rule.id, None, f"Escalated by model: {reason}"), None
    start = clause.find(quote) if quote else -1
    if start < 0:
        return (
            Assessment(NEEDS_HUMAN, rule.id, None, "Escalated: evidence quote not in clause."),
            None,
        )
    proposed = (data.get("proposed_text") or "").strip() or None
    if risk == NON_COMPLIANT and (proposed is None or rule.approved_language not in proposed):
        proposed = None  # fall back to the approved playbook language verbatim
    assessment = Assessment(risk, rule.id, "llm", reason, start, start + len(quote))
    return assessment, proposed if risk == NON_COMPLIANT else None


def make_backend(settings: Settings) -> Backend:
    if settings.backend == "llm":
        return LLMBackend(settings)
    return OfflineBackend()
