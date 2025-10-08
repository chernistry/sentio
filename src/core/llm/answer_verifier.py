from __future__ import annotations

"""Lightweight answer verification using LLM with strict JSON output.

This module audits an answer against the numbered context. If issues are found,
it can suggest a minimally revised answer. Inspired by verification patterns
observed in Navan's project (self-check plus receipts), adapted for Python.
"""

import json
import logging
from typing import Any, TypedDict

from src.core.llm.chat_adapter import ChatAdapter
from src.core.llm.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class VerifyResult(TypedDict, total=False):
    verdict: str  # pass | warn | fail
    citations_ok: bool
    notes: list[str]
    revised_answer: str


class AnswerVerifier:
    def __init__(self, chat_adapter: ChatAdapter | None = None) -> None:
        self._adapter = chat_adapter or ChatAdapter()
        self._builder = PromptBuilder()

    async def verify(self, *, query: str, context: str, answer: str) -> VerifyResult:
        """Run verification and return parsed JSON result.

        Never raises on model failure; returns a conservative default instead.
        """
        prompt = self._builder.build_verify_prompt(query=query, context=context, answer=answer)
        payload = {
            "messages": [
                {"role": "system", "content": "You are a strict verifier that outputs only JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        try:
            resp = await self._adapter.chat_completion(payload)
            text = self._extract_text(resp)
            result = self._parse_json(text)
            return result
        except Exception as exc:
            logger.warning("verify() failed: %s", exc)
            return VerifyResult(verdict="warn", citations_ok=False, notes=["verifier_error"])  # type: ignore

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        try:
            return (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        except Exception:
            return ""

    @staticmethod
    def _parse_json(text: str) -> VerifyResult:
        # Find first JSON object in text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            snippet = text[start : end + 1]
            try:
                data = json.loads(snippet)
                # Basic normalization
                out: VerifyResult = VerifyResult()
                out["verdict"] = str(data.get("verdict", "warn"))
                out["citations_ok"] = bool(data.get("citations_ok", False))
                notes = data.get("notes") or []
                out["notes"] = [str(n) for n in notes][:8]
                if "revised_answer" in data and isinstance(data["revised_answer"], str):
                    out["revised_answer"] = data["revised_answer"]
                return out
            except Exception:
                pass
        return VerifyResult(verdict="warn", citations_ok=False, notes=["invalid_json"])  # type: ignore

