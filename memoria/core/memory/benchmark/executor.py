"""Execute benchmark scenarios against a live Memoria API.

The executor:
1. Creates an isolated session per scenario
2. Loads seed memories into the system
3. Runs scenario steps (store, correct, purge, etc.)
4. Runs assertion queries and collects raw results
5. Returns raw results for the scorer to evaluate against ground truth
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from memoria.core.memory.benchmark.schema import (
    MemoryAssertion,
    Scenario,
    ScenarioStep,
)


@dataclass
class StepResult:
    action: str
    success: bool
    error: str | None = None


@dataclass
class AssertionResult:
    query: str
    returned_contents: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class ScenarioExecution:
    """Raw execution output — no scoring yet."""

    scenario_id: str
    step_results: list[StepResult] = field(default_factory=list)
    assertion_results: list[AssertionResult] = field(default_factory=list)
    error: str | None = None


class BenchmarkExecutor:
    """Executes scenarios against the Memoria REST API."""

    def __init__(
        self,
        api_url: str,
        api_token: str,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._run_id = str(int(time.time()))
        self._owned = client is None
        self._client = client or httpx.Client(
            base_url=api_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=timeout,
            trust_env=False,
        )

    def close(self) -> None:
        if self._owned:
            self._client.close()

    def execute(self, scenario: Scenario) -> ScenarioExecution:
        # Use a unique run-scoped session to prevent cross-run contamination
        session_id = f"bench-{self._run_id}-{scenario.scenario_id.lower()}"
        execution = ScenarioExecution(scenario_id=scenario.scenario_id)
        try:
            # Phase 1: load seed memories
            for seed in scenario.seed_memories:
                self._store(seed.content, seed.memory_type, session_id)

            # Phase 2: run steps
            for step in scenario.steps:
                step_result = self._run_step(step, session_id)
                execution.step_results.append(step_result)

            # Phase 3: run assertion queries
            for assertion in scenario.assertions:
                assertion_result = self._run_assertion(assertion, session_id)
                execution.assertion_results.append(assertion_result)
        except Exception as e:
            execution.error = str(e)
        return execution

    def _store(self, content: str, memory_type: str, session_id: str) -> bool:
        resp = self._client.post(
            "/v1/memories",
            json={
                "content": content,
                "memory_type": memory_type,
                "session_id": session_id,
                "source": "benchmark",
            },
        )
        resp.raise_for_status()
        return True

    def _retrieve(
        self, query: str, session_id: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        resp = self._client.post(
            "/v1/memories/retrieve",
            json={
                "query": query,
                "top_k": top_k,
                "session_id": session_id,
                "include_cross_session": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    def _search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        resp = self._client.post(
            "/v1/memories/search",
            json={"query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    def _correct(self, query: str, new_content: str, reason: str) -> bool:
        resp = self._client.post(
            "/v1/memories/correct",
            json={"query": query, "new_content": new_content, "reason": reason},
        )
        return resp.status_code < 400

    def _purge(self, memory_ids: list[str], reason: str) -> bool:
        resp = self._client.post(
            "/v1/memories/purge",
            json={"memory_ids": memory_ids, "reason": reason},
        )
        return resp.status_code < 400

    def _run_step(self, step: ScenarioStep, session_id: str) -> StepResult:
        try:
            if step.action == "store":
                self._store(
                    step.content or "", step.memory_type or "semantic", session_id
                )
                return StepResult(action="store", success=True)
            elif step.action == "retrieve":
                self._retrieve(step.query or "", session_id, step.top_k or 5)
                return StepResult(action="retrieve", success=True)
            elif step.action == "search":
                self._search(step.query or "", step.top_k or 10)
                return StepResult(action="search", success=True)
            elif step.action == "correct":
                ok = self._correct(
                    query=step.query or "",
                    new_content=step.content or "",
                    reason=step.reason or "benchmark correction",
                )
                return StepResult(action="correct", success=ok)
            elif step.action == "purge":
                ok = self._purge(
                    step.memory_ids or [], reason=step.reason or "benchmark purge"
                )
                return StepResult(action="purge", success=ok)
            else:
                return StepResult(
                    action=step.action, success=False, error="unknown action"
                )
        except Exception as e:
            return StepResult(action=step.action, success=False, error=str(e))

    def _run_assertion(
        self, assertion: MemoryAssertion, session_id: str
    ) -> AssertionResult:
        try:
            items = self._retrieve(assertion.query, session_id, assertion.top_k)
            contents = [item.get("content", "") for item in items]
            return AssertionResult(query=assertion.query, returned_contents=contents)
        except Exception as e:
            return AssertionResult(query=assertion.query, error=str(e))
