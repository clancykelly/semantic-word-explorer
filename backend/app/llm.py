"""Optional LLM enrichment via the Claude API.

Adds human-readable labels to the embedding-based clusters (e.g. turning a raw
"sea, waters..." label into "bodies of water"). Entirely optional: if
ANTHROPIC_API_KEY is unset or the `anthropic` SDK is missing, enrichment is a
no-op and the provider's heuristic labels are used unchanged.

Model defaults to claude-opus-4-8; override with ENRICH_MODEL (e.g.
claude-haiku-4-5 for a cheaper, faster per-word call).
"""

from __future__ import annotations

import json
import os

# Structured-output schema: one label per cluster id.
_SCHEMA = {
    "type": "object",
    "properties": {
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "label": {"type": "string"},
                },
                "required": ["id", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clusters"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You label clusters of related words for a thesaurus UI. Given a query word and "
    "clusters of words grouped by meaning, return a short, human-readable label "
    "(2-4 words, lowercase) for each cluster that captures what its words share — "
    "e.g. 'bodies of water', 'marine life', 'financial institutions'. "
    "Return exactly one entry per cluster id provided."
)


class LLMEnricher:
    """Labels embedding clusters via Claude. A no-op when no API key is configured."""

    def __init__(self, model: str | None = None):
        self._cache: dict[str, dict[int, str]] = {}
        self.enabled = False
        self._client = None
        self.model = model or os.getenv("ENRICH_MODEL", "claude-opus-4-8")

        if not os.getenv("ANTHROPIC_API_KEY"):
            print("LLM enrichment disabled (set ANTHROPIC_API_KEY to enable)")
            return
        try:
            from anthropic import Anthropic

            self._client = Anthropic()
            self.enabled = True
            print(f"LLM enrichment enabled (model: {self.model})")
        except Exception as e:  # noqa: BLE001 - degrade gracefully if SDK unavailable
            print(f"LLM enrichment unavailable ({e})")

    def label_clusters(
        self, query_word: str, clusters: dict[int, list[str]]
    ) -> dict[int, str] | None:
        """Return {cluster_id: label}, or None if enrichment is unavailable/failed.

        Cached per (query word + cluster membership) so each distinct result set
        hits the API at most once.
        """
        if not self.enabled or not clusters:
            return None

        cache_key = query_word + "||" + "|".join(
            f"{cid}:{','.join(words[:6])}" for cid, words in sorted(clusters.items())
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        lines = [f"Query word: {query_word}", "", "Clusters:"]
        for cid, words in sorted(clusters.items()):
            lines.append(f"- cluster {cid}: {', '.join(words[:12])}")
        prompt = "\n".join(lines)

        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "effort": "low",
                    "format": {"type": "json_schema", "schema": _SCHEMA},
                },
            )
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            )
            data = json.loads(text)
            labels = {int(c["id"]): str(c["label"]) for c in data.get("clusters", [])}
        except Exception as e:  # noqa: BLE001 - never break search on enrichment failure
            print(f"LLM enrichment error: {e}")
            return None

        self._cache[cache_key] = labels
        return labels
