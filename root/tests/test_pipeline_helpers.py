from root.src.core.pipeline import SentioRAGPipeline, GenerationMode


def test_pipeline_prompt_and_context():  # noqa: D401
    """Verify helper methods produce the expected markers in output."""

    pipeline = SentioRAGPipeline()

    sources = [
        {"text": "sentio docs", "source": "s1", "score": 0.8},
        {"text": "more docs", "source": "s2", "score": 0.7},
    ]
    ctx_str = pipeline._build_context_string(sources)  # noqa: SLF001
    assert "Source 1" in ctx_str and "Source 2" in ctx_str

    prompt_cfg = pipeline._generation_configs[GenerationMode.FAST]  # noqa: SLF001
    prompt = pipeline._build_prompt("What is Sentio?", ctx_str, prompt_cfg)

    assert "Question:" in prompt and "Context:" in prompt and "What is Sentio?" in prompt 