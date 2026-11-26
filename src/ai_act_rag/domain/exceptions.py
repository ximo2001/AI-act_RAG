"""Domain-level exceptions with explicit meaning."""


class AIActRAGError(Exception):
    """Base class for all errors raised by this package."""


class ParsingError(AIActRAGError):
    """The source document does not have the expected structure."""


class IndexNotBuiltError(AIActRAGError):
    """A retriever or pipeline was used before the index was built/loaded."""


class LLMOutputError(AIActRAGError):
    """The LLM output could not be parsed into the expected JSON schema."""


class ProviderUnavailableError(AIActRAGError):
    """An LLM/embedding provider is not installed or not reachable."""
