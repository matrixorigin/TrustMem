"""Input validation utilities."""

import re
import unicodedata
from typing import Any

from pydantic import BaseModel, Field, field_validator


def validate_identifier(name: str, max_length: int = 64, allow_dot: bool = False) -> str:
    """Validate SQL identifier (database, table, column name) to prevent SQL injection.
    
    Args:
        name: Identifier to validate
        max_length: Maximum allowed length
        allow_dot: Allow dots in name (for qualified names like db.table)
        
    Returns:
        Validated identifier
        
    Raises:
        ValueError: If identifier is invalid
        
    Example:
        >>> validate_identifier("my_table")
        'my_table'
        >>> validate_identifier("dev_agent.events", allow_dot=True)
        'dev_agent.events'
        >>> validate_identifier("'; DROP TABLE auth_users--")
        ValueError: Invalid identifier
    """
    if not name:
        raise ValueError("Identifier cannot be empty")
    
    if len(name) > max_length:
        raise ValueError(f"Identifier too long (max {max_length} characters)")
    
    # Allow alphanumeric, underscore, and optionally dot
    pattern = r'^[a-zA-Z0-9_.]+$' if allow_dot else r'^[a-zA-Z0-9_]+$'
    if not re.match(pattern, name):
        raise ValueError(f"Invalid identifier: {name}. Only alphanumeric characters, underscores{', and dots' if allow_dot else ''} are allowed")
    
    # Must start with letter or underscore
    if not name[0].isalpha() and name[0] != '_':
        raise ValueError(f"Identifier must start with letter or underscore: {name}")
    
    return name


class QueryRequest(BaseModel):
    """Query request with validation."""

    user_id: str = Field(..., min_length=1, max_length=255)
    session_id: str = Field(..., min_length=1, max_length=255)
    query: str = Field(..., min_length=1, max_length=10000)
    repo_id: int | None = Field(default=None, ge=1)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query content."""
        # Remove excessive whitespace
        v = " ".join(v.split())

        # Check for SQL injection patterns (basic)
        dangerous_patterns = [
            r";\s*DROP\s+TABLE",
            r";\s*DELETE\s+FROM",
            r";\s*UPDATE\s+.*\s+SET",
            r"UNION\s+SELECT",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Query contains potentially dangerous content")

        return v


class SkillExecutionRequest(BaseModel):
    """Skill execution request with validation."""

    skill_name: str = Field(..., min_length=1, max_length=255, pattern=r"^[a-z_][a-z0-9_]*$")
    parameters: dict[str, Any] = Field(default_factory=dict)
    user_id: str = Field(..., min_length=1, max_length=255)
    session_id: str = Field(..., min_length=1, max_length=255)

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: dict) -> dict:
        """Validate parameters size."""
        # Limit parameter size to prevent DoS
        import json

        param_size = len(json.dumps(v))
        if param_size > 100000:  # 100KB
            raise ValueError("Parameters too large (max 100KB)")
        return v


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """Sanitize string input.

    Args:
        value: Input string
        max_length: Maximum allowed length

    Returns:
        Sanitized string
    """
    # Truncate
    value = value[:max_length]

    # Remove null bytes
    value = value.replace("\x00", "")

    # Remove control characters except newline and tab
    value = "".join(
        char for char in value if char in "\n\t" or unicodedata.category(char)[0] != "C"
    )

    return value


def validate_repo_id(repo_id: int) -> int:
    """Validate repository ID.

    Args:
        repo_id: Repository ID

    Returns:
        Validated repo ID

    Raises:
        ValueError: If invalid
    """
    if repo_id < 1:
        raise ValueError("Repository ID must be positive")
    if repo_id > 2147483647:  # Max INT
        raise ValueError("Repository ID too large")
    return repo_id


def validate_session_id(session_id: str) -> str:
    """Validate session ID format.

    Args:
        session_id: Session ID

    Returns:
        Validated session ID

    Raises:
        ValueError: If invalid
    """
    # Allow alphanumeric, dash, underscore
    if not re.match(r"^[a-zA-Z0-9_-]+$", session_id):
        raise ValueError("Invalid session ID format")

    if len(session_id) > 255:
        raise ValueError("Session ID too long")

    return session_id
