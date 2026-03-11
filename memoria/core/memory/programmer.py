"""MemoryProgrammer — declarative memory manipulation via structured scripts.

Thin orchestrator over MemoryEditor + MemoryExperimentManager.
Parses YAML/dict scripts, validates actions, executes in sandbox.

See docs/design/memory/backend-management.md §9
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from memoria.core.db_consumer import DbFactory
    from memoria.core.memory.editor import MemoryEditor
    from memoria.core.memory.experiment import ExperimentInfo, MemoryExperimentManager


logger = logging.getLogger(__name__)

SCRIPT_VERSION = 1


# ── Action schemas ────────────────────────────────────────────────────


class InjectAction(BaseModel):
    """Inject a new memory."""

    inject: dict = Field(...)

    # Nested fields
    @property
    def content(self) -> str:
        return self.inject["content"]

    @property
    def memory_type(self) -> str:
        return self.inject.get("type", "semantic")

    @property
    def trust_tier(self) -> str:
        return self.inject.get("trust", "T2")


class CorrectAction(BaseModel):
    """Correct an existing memory."""

    correct: dict = Field(...)

    @property
    def memory_id(self) -> str:
        return self.correct["memory_id"]

    @property
    def new_content(self) -> str:
        return self.correct["new_content"]


class PurgeAction(BaseModel):
    """Purge memories matching filter."""

    purge: dict = Field(...)

    @property
    def filter_spec(self) -> dict:
        return self.purge.get("filter", {})


class TuneAction(BaseModel):
    """Tune strategy params."""

    tune: dict = Field(...)

    @property
    def strategy(self) -> str:
        return self.tune["strategy"]

    @property
    def params(self) -> dict:
        return self.tune.get("params", {})


# Action type → key used in script
_ACTION_KEYS = {"inject", "correct", "purge", "tune"}


class InvalidScriptError(ValueError):
    """Raised when a memory program script is invalid."""


class ProgramTimeoutError(TimeoutError):
    """Raised when a memory program exceeds its timeout."""


@dataclass
class ActionResult:
    """Result of a single action execution."""

    action_type: str
    success: bool
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ProgramResult:
    """Result of executing a memory program."""

    experiment_id: str | None = None
    actions_executed: int = 0
    actions_failed: int = 0
    results: list[ActionResult] = field(default_factory=list)
    dry_run: bool = False
    rolled_back: bool = False
    timed_out: bool = False


def parse_script(raw: str | dict | list) -> list[dict]:
    """Parse a memory program script into a list of action dicts.

    Accepts:
    - dict with 'actions' key (full script format)
    - list of action dicts
    - YAML string

    Returns:
        List of action dicts, each with exactly one action key.

    Raises:
        InvalidScriptError: If script is malformed.
    """
    if isinstance(raw, str):
        import re

        import yaml
        # Strip markdown code fences (```yaml ... ``` or ``` ... ```)
        raw = re.sub(r"^```(?:ya?ml)?\s*\n", "", raw.strip())
        raw = re.sub(r"\n```\s*$", "", raw)
        try:
            raw = yaml.safe_load(raw)
        except Exception as e:
            raise InvalidScriptError(f"Invalid YAML: {e}") from e

    if isinstance(raw, dict):
        if "actions" in raw:
            version = raw.get("version", 1)
            if int(version) != SCRIPT_VERSION:
                raise InvalidScriptError(
                    f"Unsupported script version {version} (expected {SCRIPT_VERSION})"
                )
            actions = raw["actions"]
        else:
            # Single action dict
            actions = [raw]
    elif isinstance(raw, list):
        actions = raw
    else:
        raise InvalidScriptError(f"Expected dict, list, or YAML string, got {type(raw).__name__}")

    if not actions:
        raise InvalidScriptError("Script has no actions")

    # Validate each action has exactly one known key
    for i, action in enumerate(actions):
        if not isinstance(action, dict):
            raise InvalidScriptError(f"Action {i} is not a dict")
        # Normalize flat format: {action: "inject", content: ...} → {inject: {content: ...}}
        if "action" in action and action["action"] in _ACTION_KEYS:
            action_type = action.pop("action")
            actions[i] = {action_type: action}
            action = actions[i]
        keys = set(action.keys()) & _ACTION_KEYS
        if len(keys) == 0:
            raise InvalidScriptError(
                f"Action {i} has no recognized action key (expected one of {_ACTION_KEYS})"
            )
        if len(keys) > 1:
            raise InvalidScriptError(f"Action {i} has multiple action keys: {keys}")

    return [_normalize_action_fields(a) for a in actions]


# ── Field name normalization ─────────────────────────────────────────
# LLMs output varying field names. Map all known variants to canonical names
# so the executor only deals with one vocabulary.

_FIELD_ALIASES: dict[str, str] = {
    # inject / correct fields
    "memory_type": "type",
    "kind": "type",
    "trust_tier": "trust",
    "confidence_level": "trust",
    "tier": "trust",
    "text": "content",
    "message": "content",
    "body": "content",
    "new_text": "new_content",
    "new_message": "new_content",
    "updated_content": "new_content",
    # tune fields
    "strategy_key": "strategy",
    "strategy_name": "strategy",
    # purge fields — "filter" is already canonical
}


def _normalize_action_fields(action: dict) -> dict:
    """Normalize field names inside each action's spec dict."""
    normalized = {}
    for key, spec in action.items():
        if key in _ACTION_KEYS and isinstance(spec, dict):
            normalized[key] = _remap_fields(spec)
        else:
            normalized[key] = spec
    return normalized


def _remap_fields(spec: dict) -> dict:
    """Remap alias field names to canonical names, recursing into nested dicts."""
    out: dict = {}
    for k, v in spec.items():
        canonical = _FIELD_ALIASES.get(k, k)
        if canonical not in out:
            out[canonical] = _remap_fields(v) if isinstance(v, dict) else v
    return out


def _get_action_type(action: dict) -> str:
    """Get the action type key from an action dict."""
    for key in _ALL_ACTION_KEYS:
        if key in action:
            return key
    raise InvalidScriptError(f"No action key found in {action}")


_BATCH_INJECT_KEY = "_batch_inject"
_ALL_ACTION_KEYS = _ACTION_KEYS | {_BATCH_INJECT_KEY}


def _coalesce_injects(actions: list[dict]) -> list[dict]:
    """Merge consecutive inject actions into batch inserts.

    [inject, inject, inject, purge, inject] →
    [_batch_inject(3), purge, inject(1)]
    """
    result: list[dict] = []
    batch: list[dict] = []

    def flush() -> None:
        if len(batch) == 1:
            result.append({"inject": batch[0]})
        elif len(batch) > 1:
            result.append({_BATCH_INJECT_KEY: batch[:]})
        batch.clear()

    for action in actions:
        if "inject" in action:
            batch.append(action["inject"])
        else:
            flush()
            result.append(action)
    flush()
    return result


class MemoryProgrammer:
    """Declarative memory manipulation via structured scripts.

    Composes MemoryEditor (actions) + MemoryExperimentManager (sandbox).

    sandbox default: False in the CLI EdgeTool, True in the core API.
    The core API defaults to sandbox=True for safety (atomic rollback on
    failure, experiment branch isolation). The CLI EdgeTool overrides this
    to False because no commit UI exists yet — sandbox writes would silently
    disappear since the LLM cannot trigger the commit step.
    """

    def __init__(
        self,
        editor: MemoryEditor,
        experiments: MemoryExperimentManager,
        db_factory: DbFactory,
    ) -> None:
        self._editor = editor
        self._experiments = experiments
        self._db_factory = db_factory

    def execute(
        self,
        user_id: str,
        script: str | dict | list,
        *,
        sandbox: bool = True,
        dry_run: bool = False,
        atomic: bool = True,
        timeout_seconds: float | None = None,
        program_name: str = "unnamed",
        session_id: str | None = None,
    ) -> ProgramResult:
        """Parse and execute a memory program.

        Args:
            user_id: Target user.
            script: YAML string, dict, or list of actions.
            sandbox: If True (default), execute in experiment branch.
            dry_run: If True, parse and validate only, don't execute.
            atomic: If True (default), discard on any failure (sandbox)
                    or stop-on-first-failure (non-sandbox).
            timeout_seconds: Max wall-clock seconds for execution. None = no limit.
            program_name: Name for the experiment (if sandboxed).

        Returns:
            ProgramResult with per-action results.

        Raises:
            InvalidScriptError: If script is malformed.
            ProgramTimeoutError: If execution exceeds timeout_seconds.
        """
        import time

        actions = parse_script(script)

        if dry_run:
            return ProgramResult(
                actions_executed=0,
                results=[
                    ActionResult(action_type=_get_action_type(a), success=True,
                                 detail={"dry_run": True})
                    for a in actions
                ],
                dry_run=True,
            )

        # Batch consecutive injects for performance
        actions = _coalesce_injects(actions)

        exp_info: ExperimentInfo | None = None
        editor = self._editor
        if sandbox:
            exp_info = self._experiments.create(
                user_id, f"prog_{program_name}",
                description=f"Memory program: {program_name}",
            )
            editor = self._make_branch_editor(exp_info)

        deadline = (time.monotonic() + timeout_seconds) if timeout_seconds else None
        results: list[ActionResult] = []
        failed_any = False
        timed_out = False
        for action in actions:
            if atomic and failed_any:
                break
            if deadline and time.monotonic() >= deadline:
                timed_out = True
                break
            result = self._execute_action(user_id, action, editor=editor, session_id=session_id)
            results.append(result)
            if not result.success:
                failed_any = True

        executed = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)

        # Atomic rollback: discard experiment on any failure or timeout
        if atomic and (failed_any or timed_out) and sandbox and exp_info:
            self._experiments.discard(exp_info.experiment_id)
            pr = ProgramResult(
                experiment_id=exp_info.experiment_id,
                actions_executed=executed,
                actions_failed=failed,
                results=results,
                rolled_back=True,
                timed_out=timed_out,
            )
            self._log_program_audit(user_id, program_name, pr)
            return pr

        if timed_out:
            raise ProgramTimeoutError(
                f"Execution timed out after {timeout_seconds}s "
                f"({len(results)}/{len(actions)} actions completed)"
            )

        pr = ProgramResult(
            experiment_id=exp_info.experiment_id if exp_info else None,
            actions_executed=executed,
            actions_failed=failed,
            results=results,
        )
        self._log_program_audit(user_id, program_name, pr)
        return pr

    def _log_program_audit(
        self, user_id: str, program_name: str, result: ProgramResult,
    ) -> None:
        """Write a program-level entry to mem_edit_log.

        For sandbox runs, also writes per-action inject/correct/purge entries to
        production DB so the audit trail is complete after commit.
        """
        import json

        from sqlalchemy import text

        from memoria.core.utils.id_generator import generate_id

        memory_ids = []
        for r in result.results:
            if r.success and r.detail:
                if "memory_id" in r.detail:
                    memory_ids.append(r.detail["memory_id"])
                elif "memory_ids" in r.detail:
                    memory_ids.extend(r.detail["memory_ids"])
        try:
            with self._db_factory() as db:
                # For sandbox runs: mirror per-action audit entries to production DB.
                # Branch editor writes them to branch DB only; they're lost after commit.
                if result.experiment_id:
                    for r in result.results:
                        if not r.success or r.action_type not in ("inject", "correct", "purge"):
                            continue
                        action_ids: list[str] = []
                        if r.detail:
                            if "memory_id" in r.detail:
                                action_ids = [r.detail["memory_id"]]
                            elif "memory_ids" in r.detail:
                                action_ids = list(r.detail["memory_ids"])
                        db.execute(
                            text(
                                "INSERT INTO mem_edit_log "
                                "(edit_id, user_id, operation, target_ids, reason, "
                                " snapshot_before, created_by) "
                                "VALUES (:eid, :uid, :op, :tids, :reason, :snap, :uid)"
                            ),
                            {
                                "eid": generate_id(),
                                "uid": user_id,
                                "op": r.action_type,
                                "tids": json.dumps(action_ids),
                                "reason": f"sandbox:{program_name}",
                                "snap": result.experiment_id,
                            },
                        )

                db.execute(
                    text(
                        "INSERT INTO mem_edit_log "
                        "(edit_id, user_id, operation, target_ids, reason, "
                        " snapshot_before, created_by) "
                        "VALUES (:eid, :uid, :op, :tids, :reason, :snap, :uid)"
                    ),
                    {
                        "eid": generate_id(),
                        "uid": user_id,
                        "op": "program",
                        "tids": json.dumps(memory_ids),
                        "reason": program_name,
                        "snap": result.experiment_id,
                    },
                )
                db.commit()
        except Exception:
            logger.debug("Failed to log program audit for %s", user_id, exc_info=True)

    def _make_branch_editor(self, exp_info: ExperimentInfo) -> MemoryEditor:
        """Create a MemoryEditor that operates on the experiment's branch DB."""
        from memoria.core.memory.canonical_storage import CanonicalStorage
        from memoria.core.memory.editor import MemoryEditor as EditorCls

        branch_factory = self._experiments._make_branch_db_factory(
            exp_info.branch_db, exp_info.experiment_id,
        )
        storage = CanonicalStorage(branch_factory)
        return EditorCls(storage, branch_factory)

    def _execute_action(self, user_id: str, action: dict, *, editor: MemoryEditor, session_id: str | None = None) -> ActionResult:
        """Execute a single action via MemoryEditor."""
        raw_type = _get_action_type(action)
        display_type = "inject" if raw_type == _BATCH_INJECT_KEY else raw_type
        try:
            if raw_type == "inject":
                return self._do_inject(user_id, action["inject"], editor=editor, session_id=session_id)
            if raw_type == _BATCH_INJECT_KEY:
                return self._do_batch_inject(user_id, action[_BATCH_INJECT_KEY], editor=editor, session_id=session_id)
            if raw_type == "correct":
                return self._do_correct(user_id, action["correct"], editor=editor)
            if raw_type == "purge":
                return self._do_purge(user_id, action["purge"], editor=editor)
            if raw_type == "tune":
                return self._do_tune(user_id, action["tune"])
            return ActionResult(action_type=display_type, success=False,
                                error=f"Unknown action: {display_type}")
        except Exception as e:
            return ActionResult(action_type=display_type, success=False, error=str(e))

    def _do_batch_inject(
        self, user_id: str, specs: list[dict], *, editor: MemoryEditor, session_id: str | None = None,
    ) -> ActionResult:
        """Batch-insert multiple memories via editor.batch_inject (single transaction)."""
        # Validate all specs before calling editor
        for spec in specs:
            if not spec.get("content"):
                return ActionResult(
                    action_type="inject", success=False,
                    error="inject requires 'content'",
                )

        stored = editor.batch_inject(user_id, specs, source="batch_inject", session_id=session_id)
        return ActionResult(
            action_type="inject", success=True,
            detail={"memory_ids": [m.memory_id for m in stored], "count": len(stored)},
        )

    def _do_inject(self, user_id: str, spec: dict, *, editor: MemoryEditor, session_id: str | None = None) -> ActionResult:
        from memoria.core.memory.types import MemoryType, TrustTier

        content = spec.get("content")
        if not content:
            return ActionResult(action_type="inject", success=False,
                                error="inject requires 'content'")

        # Coerce LLM-friendly type aliases to valid MemoryType
        _TYPE_ALIASES = {"preference": "profile", "fact": "semantic", "skill": "procedural"}
        raw_type = spec.get("type", "semantic")
        mem_type = MemoryType(_TYPE_ALIASES.get(raw_type, raw_type))

        # Coerce numeric trust to TrustTier (LLMs often send floats)
        raw_trust = spec.get("trust", "T2")
        if isinstance(raw_trust, (int, float)):
            raw_trust = "T1" if raw_trust >= 0.9 else "T2" if raw_trust >= 0.7 else "T3" if raw_trust >= 0.4 else "T4"
        trust = TrustTier(raw_trust)

        mem = editor.inject(
            user_id, content,
            memory_type=mem_type,
            trust_tier=trust,
            session_id=session_id,
        )
        return ActionResult(
            action_type="inject", success=True,
            detail={"memory_id": mem.memory_id},
        )

    def _do_correct(self, user_id: str, spec: dict, *, editor: MemoryEditor) -> ActionResult:
        memory_id = spec.get("memory_id")
        new_content = spec.get("new_content")
        if not memory_id or not new_content:
            return ActionResult(action_type="correct", success=False,
                                error="correct requires 'memory_id' and 'new_content'")

        mem = editor.correct(
            user_id, memory_id, new_content,
            reason=spec.get("reason", ""),
        )
        return ActionResult(
            action_type="correct", success=True,
            detail={"old_id": memory_id, "new_id": mem.memory_id},
        )

    def _do_purge(self, user_id: str, spec: dict, *, editor: MemoryEditor) -> ActionResult:
        from datetime import datetime, timezone

        from memoria.core.memory.types import MemoryType

        filter_spec = spec.get("filter", {})
        memory_ids = filter_spec.get("memory_ids")
        type_val = filter_spec.get("type")
        memory_types = [MemoryType(type_val)] if type_val else None

        before: datetime | None = None
        if before_str := filter_spec.get("before"):
            before = datetime.fromisoformat(before_str).replace(tzinfo=timezone.utc) \
                if datetime.fromisoformat(before_str).tzinfo is None \
                else datetime.fromisoformat(before_str)

        result = editor.purge(
            user_id,
            memory_ids=memory_ids,
            memory_types=memory_types,
            before=before,
            reason=spec.get("reason", ""),
        )
        return ActionResult(
            action_type="purge", success=True,
            detail={"deactivated": result.deactivated,
                    "snapshot": result.snapshot_name},
        )

    def _do_tune(self, user_id: str, spec: dict) -> ActionResult:
        from memoria.core.memory.strategy.params import validate_strategy_params

        strategy = spec.get("strategy")
        params = spec.get("params")
        if not strategy:
            return ActionResult(action_type="tune", success=False,
                                error="tune requires 'strategy'")

        validated = validate_strategy_params(strategy, params)

        from memoria.core.memory.factory import set_user_strategy

        set_user_strategy(self._db_factory, user_id, strategy)

        # Update params in user config if provided
        if validated:
            from sqlalchemy import func as sa_func

            from memoria.core.memory.models.memory_config import MemoryUserConfig

            with self._db_factory() as db:
                db.query(MemoryUserConfig).filter_by(
                    user_id=user_id,
                ).update({"params_json": validated, "updated_at": sa_func.now()})
                db.commit()

        return ActionResult(
            action_type="tune", success=True,
            detail={"strategy": strategy, "params": validated},
        )


def nl_to_script(user_input: str, user_id: str, llm_client: Any, *, model: str | None = None) -> list[dict]:
    """Convert natural language instruction to structured actions via LLM.

    Args:
        user_input: Natural language memory instruction.
        user_id: Target user.
        llm_client: LLMClient instance with .chat() method.
        model: LLM model to use (default: caller's default).

    Returns:
        Parsed list of action dicts.

    Raises:
        InvalidScriptError: If LLM output is not valid YAML/script.
    """
    from memoria.core.memory.programmer_prompts import NL_TO_SCRIPT_PROMPT

    prompt = NL_TO_SCRIPT_PROMPT.format(user_input=user_input, user_id=user_id)
    kwargs: dict = {
        "messages": [{"role": "user", "content": prompt}],
        "user_id": user_id,
        "task_hint": "memory_program_nl_convert",
        "temperature": 0.0,
        "model": model or "cheapest",
    }
    response = llm_client.chat(**kwargs)
    return parse_script(response.content)
