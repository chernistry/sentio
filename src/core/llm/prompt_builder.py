"""Utility class to centralise prompt construction logic.

This small helper loads template files from ``prompts`` directory once and formats
those templates with runtime placeholders. Keeping all prompt-related logic
in a single place avoids subtle divergences between modules and makes prompt
engineering iterations significantly easier.

The builder is intentionally dependency-free so it can be used from anywhere
(e.g. RAG pipeline, CLI tools, unit tests).
"""

from pathlib import Path
from typing import Final, Literal, cast

__all__: Final = ["PromptBuilder"]

_PROMPT_DIR: Final = Path(__file__).resolve().parents[3] / "prompts"

_Mode = Literal["fast", "balanced", "quality", "creative"]


class PromptBuilder:
    """Simple loader + formatter for prompt templates.

    The class keeps the raw template strings in memory after the first load to
    minimise IO overhead when the pipeline is serving many concurrent
    requests.
    """

    _retrieve_template: str | None = None
    _summarise_template: str | None = None
    _verify_template: str | None = None
    _profile_template: str | None = None

    _MODE_INSTRUCTIONS: dict[_Mode, str] = {
        "fast": "Provide a concise, direct answer.",
        "balanced": "Provide a comprehensive but focused answer.",
        "quality": "Provide a detailed, well-structured answer with proper explanations.",
        "creative": "Provide an engaging, creative answer while staying factual.",
    }

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def build_generation_prompt(self, query: str, context: str, mode: _Mode = "balanced") -> str:
        """Return the fully-formatted generation prompt.

        Args:
            query: End-user question.
            context: Retrieved context string (may be empty).
            mode: One of the predefined generation modes.
            
        Returns:
            Formatted prompt text
        """
        template = self._get_retrieve_template()
        instruction = self._MODE_INSTRUCTIONS.get(mode, self._MODE_INSTRUCTIONS["balanced"])
        return (
            template.replace("{instruction}", instruction)
            .replace("{context}", context)
            .replace("{query}", query)
        )

    def build_system_message(self) -> str:
        """Return the system message for chat models.
        
        Returns:
            System message text
        """
        return self._get_profile_template()

    def build_verify_prompt(self, query: str, context: str, answer: str) -> str:
        """Return the verification prompt instructing the model to audit the answer.

        Args:
            query: Original user question
            context: Numbered context string used for generation
            answer: The model's answer to verify

        Returns:
            Formatted verification prompt text
        """
        template = self._get_verify_template()
        return (
            template
            .replace("{query}", query)
            .replace("{context}", context)
            .replace("{answer}", answer)
        )

    # ------------------------------------------------------------------
    # Internal template loaders (lazy)
    # ------------------------------------------------------------------

    @classmethod
    def _load_template(cls, filename: str, *, fallback: str) -> str:
        path = _PROMPT_DIR / filename
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # Fail gracefully with a minimal viable template to avoid runtime
            # crashes in production – error will be logged upstream.
            return fallback

    @classmethod
    def _get_retrieve_template(cls) -> str:
        if cls._retrieve_template is None:
            cls._retrieve_template = cls._load_template(
                "retrieve.md",
                fallback=(
                    "You are Sentio, an expert AI assistant with access to a comprehensive knowledge base.\n\n"
                    "Your task: {instruction}\n\n"
                    "Guidelines:\n"
                    "- Base your answer strictly on the provided context\n"
                    "- If the context is insufficient, clearly state what information is missing\n"
                    "- Cite sources when relevant\n"
                    "- Be honest about limitations\n"
                    "- Maintain a professional yet accessible tone\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {query}\n\n"
                    "Answer:"
                ),
            )
        return cast("str", cls._retrieve_template)

    @classmethod
    def _get_summarise_template(cls) -> str:
        if cls._summarise_template is None:
            cls._summarise_template = cls._load_template(
                "summarize.md",
                fallback="Summarise the following content:\n\n{content}\n\nSummary:",
            )
        return cast("str", cls._summarise_template)

    @classmethod
    def _get_profile_template(cls) -> str:
        if cls._profile_template is None:
            cls._profile_template = cls._load_template(
                "profile.md",
                fallback="You are Sentio, an enterprise–grade AI system.",
            )
        return cast("str", cls._profile_template)

    @classmethod
    def _get_verify_template(cls) -> str:
        if cls._verify_template is None:
            cls._verify_template = cls._load_template(
                "verify.md",
                fallback=(
                    "You are a rigorous answer verifier. Audit the answer against the numbered Context.\n\n"
                    "Output strict JSON with fields: verdict (pass|warn|fail), notes (array of strings), "
                    "citations_ok (boolean), revised_answer (optional string).\n\n"
                    "Rules:\n"
                    "- Every factual claim relying on context must include bracketed citations [n] that exist in Context.\n"
                    "- If citations are missing or unsupported, set verdict to 'fail' and provide revised_answer that fixes issues.\n"
                    "- If minor issues (style/format) exist but facts are supported, set verdict to 'warn' and include notes.\n"
                    "- If adequate and properly cited, set verdict to 'pass'.\n\n"
                    "Question: {query}\n\nContext:\n{context}\n\nAnswer:\n{answer}\n\nJSON:"
                ),
            )
        return cast("str", cls._verify_template)
