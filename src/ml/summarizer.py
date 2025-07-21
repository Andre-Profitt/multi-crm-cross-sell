import logging
from typing import Optional

_logger = logging.getLogger(__name__)
_model = None

async def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from transformers import pipeline
        _model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:
        _logger.warning("Failed to load transformers summarization model: %s", e)
        _model = None
    return _model

async def summarize_text(text: str, max_length: int = 60, min_length: int = 20) -> str:
    """Summarize a block of text. Returns empty string if model unavailable."""
    model = await _load_model()
    if not model or not text:
        return ""
    try:
        summary = model(text, max_length=max_length, min_length=min_length, truncation=True)
        return summary[0]["summary_text"]
    except Exception as e:
        _logger.error("Summarization failed: %s", e)
        return ""

async def summarize_account(account_text: str) -> str:
    """Convenience wrapper to summarize account-related text."""
    return await summarize_text(account_text)
