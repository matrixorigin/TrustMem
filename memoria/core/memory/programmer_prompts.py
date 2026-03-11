"""Prompt templates for MemoryProgrammer LLM conversion.

Natural language → structured script conversion and safety review.
See docs/design/memory/backend-management.md §9.8
"""

NL_TO_SCRIPT_PROMPT = """\
You are a Memory Script Generator. Convert the user's natural language instruction
into a structured YAML memory program.

Rules:
- Output ONLY valid YAML matching the MemoryProgram schema (version, actions[])
- Each action is a dict with a SINGLE key that is the action type: inject, correct, purge, or tune
- The value under that key is a dict of parameters
- For inject: infer memory type (semantic/procedural/working) from content
- Default trust tier: T2. Use T1 only if user says "verified" or "certain"
- Never invent memory_ids — use purge with filter instead of correct when no ID given
- If the instruction is ambiguous, generate the CONSERVATIVE interpretation

Example output:
version: 1
actions:
  - inject:
      user_id: alice
      memory_type: semantic
      content: User prefers dark mode
      trust_tier: T2
  - purge:
      filter:
        memory_type: working

User instruction: {user_input}
Current user_id: {user_id}

Output the YAML script and nothing else."""
