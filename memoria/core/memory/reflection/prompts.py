"""Reflection prompt templates.

Managed by PromptOptimizer (prompt_template_id = "reflection_synthesis").
"""

REFLECTION_SYNTHESIS_PROMPT = """\
SYSTEM:
You are analyzing an agent's experiences across multiple sessions with the same user.
Your goal: extract 1-2 reusable insights that will help the agent serve this user better.

RULES:
- Each insight must be ACTIONABLE (a behavioral rule or factual pattern), not just descriptive
- Each insight must be grounded in the evidence — do not speculate beyond what's shown
- If the experiences don't reveal a clear pattern, return an empty array []
- Assign confidence conservatively: 0.3-0.5 for weak patterns, 0.5-0.7 for strong ones
- Type "procedural" = how to do something; "semantic" = what is true about the user/project

EXISTING KNOWLEDGE (do not repeat these):
{existing_knowledge}

EXPERIENCES TO ANALYZE:
{experiences}

OUTPUT FORMAT (JSON array, 0-2 items):
[
  {{
    "type": "procedural" | "semantic",
    "content": "One clear sentence describing the insight",
    "confidence": 0.3-0.7,
    "evidence_summary": "Which experiences support this (one sentence)"
  }}
]
"""
