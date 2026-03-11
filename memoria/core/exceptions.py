"""Custom exceptions for mo-agent-engine."""


class AgentError(Exception):
    """Base exception for all agent errors."""

    def __init__(self, message: str, code: str = "AGENT_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class SkillError(AgentError):
    """Base exception for skill-related errors."""

    def __init__(self, message: str, skill_name: str | None = None):
        self.skill_name = skill_name
        super().__init__(message, code="SKILL_ERROR")


class SkillNotFoundError(SkillError):
    """Skill not found in registry."""

    hint = "Run 'diagnose_skills' to check skill health"

    def __init__(self, skill_name: str, version: str | None = None):
        message = f"Skill '{skill_name}'"
        if version:
            message += f" version '{version}'"
        message += " not found"
        super().__init__(message, skill_name=skill_name)
        self.code = "SKILL_NOT_FOUND"


class SkillExecutionError(SkillError):
    """Skill execution failed."""

    def __init__(self, skill_name: str, message: str):
        super().__init__(f"Skill '{skill_name}' execution failed: {message}", skill_name=skill_name)
        self.code = "SKILL_EXECUTION_ERROR"


class SkillValidationError(SkillError):
    """Skill input validation failed."""

    def __init__(self, skill_name: str, message: str):
        super().__init__(
            f"Skill '{skill_name}' validation failed: {message}", skill_name=skill_name
        )
        self.code = "SKILL_VALIDATION_ERROR"


class ReplayError(AgentError):
    """Replay operation failed."""

    def __init__(self, message: str, session_id: str | None = None):
        self.session_id = session_id
        super().__init__(message, code="REPLAY_ERROR")


class DatabaseError(AgentError):
    """Database operation failed."""

    def __init__(self, message: str):
        super().__init__(message, code="DATABASE_ERROR")


class ContextError(AgentError):
    """Context management error."""

    def __init__(self, message: str):
        super().__init__(message, code="CONTEXT_ERROR")


class LLMError(AgentError):
    """LLM operation failed."""

    def __init__(self, message: str, provider: str | None = None):
        self.provider = provider
        super().__init__(message, code="LLM_ERROR")


class LLMTimeoutError(LLMError):
    """LLM request timed out."""

    def __init__(self, provider: str, timeout: float):
        super().__init__(f"LLM request to {provider} timed out after {timeout}s", provider=provider)
        self.code = "LLM_TIMEOUT"


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded."""

    def __init__(self, provider: str):
        super().__init__(f"LLM rate limit exceeded for {provider}", provider=provider)
        self.code = "LLM_RATE_LIMIT"


class GitHubError(AgentError):
    """GitHub API operation failed."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message, code="GITHUB_ERROR")


class GitHubRateLimitError(GitHubError):
    """GitHub API rate limit exceeded."""

    def __init__(self):
        super().__init__("GitHub API rate limit exceeded", status_code=429)
        self.code = "GITHUB_RATE_LIMIT"


class ConfigurationError(AgentError):
    """Configuration error."""

    def __init__(self, message: str):
        super().__init__(message, code="CONFIGURATION_ERROR")


class AuthenticationError(AgentError):
    """Authentication failed."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, code="AUTHENTICATION_ERROR")


class AuthorizationError(AgentError):
    """Authorization failed."""

    def __init__(self, message: str = "Authorization failed"):
        super().__init__(message, code="AUTHORIZATION_ERROR")


class TransientError(AgentError):
    """Retryable error (DB timeout, network, rate limit)."""

    def __init__(self, message: str, retry_after_ms: int = 1000):
        self.retry_after_ms = retry_after_ms
        super().__init__(message, code="TRANSIENT_ERROR")


class MemoryError(AgentError):
    """Memory subsystem error."""

    def __init__(self, message: str):
        super().__init__(message, code="MEMORY_ERROR")


class GraphIngestError(MemoryError):
    """Graph ingest failed after tabular write succeeded (dual-write inconsistency)."""

    def __init__(self, memory_id: str, cause: Exception):
        self.memory_id = memory_id
        self.cause = cause
        super().__init__(f"Graph ingest failed for memory {memory_id}: {cause}")
