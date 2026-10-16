"""Backend-neutral LLM interface.

Backends are imported from their own modules (e.g. app.llm.ollama_client)
so that importing this package does not require every backend's SDK.
"""

from app.llm.base import LLMClient, LLMConnectionError, LLMResponse, Message

__all__ = ["LLMClient", "LLMConnectionError", "LLMResponse", "Message"]