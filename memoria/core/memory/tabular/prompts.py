"""Prompt templates for memory Observer."""

OBSERVER_EXTRACTION_PROMPT = """\
Extract structured memories from this conversation turn.
Return a JSON array ONLY, no other text. Each item:
{"type": "profile|semantic|procedural",
 "content": "concise factual statement",
 "confidence": 0.0-1.0}

Types (choose the MOST SPECIFIC type — prefer profile over semantic when applicable):
- profile: user identity, preferences, environment, habits, tools, language, role.
  Examples: "prefers Go over Python", "uses vim", "works on mo-dev-agent project",
  "speaks Chinese", "is a backend developer", "uses conda for env management",
  "prefers concise code review feedback", "runs Linux"
- semantic: general knowledge or facts learned from the conversation that are NOT
  about the user themselves. Examples: "MatrixOne supports time-travel queries",
  "event sourcing stores all state changes as events"
- procedural: repeated action patterns the user follows across multiple turns.
  Examples: "always runs tests before commit", "reviews staged changes before pushing"

IMPORTANT: If a fact describes WHO the user is, WHAT they prefer, or HOW they work,
it is "profile" — not "semantic" or "procedural".

Confidence guide:
- 1.0: user explicitly stated
- 0.7: strongly implied by context
- 0.4: weakly inferred

Do NOT extract: transient requests, greetings, meta-conversation.
If nothing worth remembering, return [].
"""
