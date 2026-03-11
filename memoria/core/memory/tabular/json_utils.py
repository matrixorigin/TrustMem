"""Utility for robustly parsing JSON arrays from LLM output.

LLMs often wrap JSON in markdown fences or include preamble text.
This module handles all common formats.
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_array(text_str: str) -> list[dict[str, Any]]:
    """Robustly extract a JSON array from LLM output.

    Handles:
    - Plain JSON array
    - JSON wrapped in ```json ... ``` fences
    - JSON embedded in surrounding prose
    """
    text_str = text_str.strip()
    try:
        result = json.loads(text_str)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\[.*?])\s*```", text_str, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\[.*]", text_str, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return []
