"""Unified ID generation utilities."""

import hashlib
import json
from typing import Any

from uuid_utils import uuid7


def generate_id(max_length: int = 36) -> str:
    """Generate a unique ID that fits within database constraints.
    
    Args:
        max_length: Maximum length for the ID (default 36 for VARCHAR(36))
        
    Returns:
        A unique ID string that fits within the specified length
    """
    # Generate UUID7 and remove hyphens
    uuid_str = str(uuid7()).replace("-", "")

    # Truncate if necessary to fit database constraints
    if len(uuid_str) > max_length:
        return uuid_str[:max_length]

    return uuid_str


def generate_hash_id(data: Any, length: int = 16) -> str:
    """Generate a deterministic hash-based ID from data.
    
    Args:
        data: Data to hash (will be JSON serialized if not string)
        length: Length of the hash ID (default 16)
        
    Returns:
        A hash-based ID string
    """
    if isinstance(data, str):
        data_str = data
    else:
        data_str = json.dumps(data, sort_keys=True)

    return hashlib.sha256(data_str.encode()).hexdigest()[:length]


def generate_display_id(full_id: str, length: int = 8) -> str:
    """Generate a short display ID from a full ID.
    
    Args:
        full_id: Full ID string
        length: Length for display (default 8)
        
    Returns:
        Truncated ID for display purposes
    """
    return full_id[:length]


# Specific ID generators for different use cases
def generate_learning_id() -> str:
    """Generate a unique learning ID for skill selection learning."""
    return generate_id(36)


def generate_event_id() -> str:
    """Generate a unique event ID."""
    return generate_id(36)


def generate_gate_id() -> str:
    """Generate a unique gate ID."""
    return generate_id(36)


def generate_note_id() -> str:
    """Generate a unique note ID."""
    return generate_id(36)


def generate_log_id() -> str:
    """Generate a unique log ID."""
    return generate_id(36)


def generate_sandbox_name(prefix: str = "sandbox") -> str:
    """Generate a unique sandbox name."""
    return f"{prefix}_{generate_id()}".lower()


def generate_prefixed_id(prefix: str, length: int = 0) -> str:
    """Generate a prefixed unique ID like 'memories_sandbox_<uuid7>'.

    Args:
        prefix: Human-readable prefix (e.g. 'memories_sandbox', 'mem_milestone')
        length: Unused, kept for backward compatibility.
    """
    return f"{prefix}_{generate_id()}"


def generate_tool_call_id() -> str:
    """Generate a tool-call ID compatible with OpenAI's format (24-char alphanum)."""
    return f"call_{generate_id(24)}"


def generate_session_name(prefix: str = "session") -> str:
    """Generate a unique session name."""
    return f"{prefix}_{generate_id()}"


def generate_test_name(prefix: str = "test") -> str:
    """Generate a unique test name."""
    return f"{prefix}_{generate_id()}"
