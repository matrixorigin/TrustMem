"""ReflectionEngine — backend-agnostic pattern synthesis.

Receives candidates from CandidateProvider → importance filter →
LLM synthesis → persist as scene-type memories.

Imports only from interfaces.py and types.py — never from tabular/ or graph/.

See docs/design/memory/backend-coexistence.md
See docs/design/memory/graph-memory.md §4.3
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from memoria.core.memory.interfaces import CandidateProvider, MemoryWriter, ReflectionCandidate
from memoria.core.memory.reflection.importance import DAILY_THRESHOLD
from memoria.core.memory.reflection.prompts import REFLECTION_SYNTHESIS_PROMPT
from memoria.core.memory.types import MemoryType, TrustTier

logger = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    """Result of a reflection cycle."""

    candidates_found: int = 0
    candidates_passed: int = 0
    scenes_created: int = 0
    llm_calls: int = 0
    errors: list[str] = field(default_factory=list)
    total_ms: float = 0.0


@dataclass
class SynthesizedInsight:
    """An insight produced by LLM synthesis."""

    memory_type: MemoryType
    content: str
    confidence: float
    evidence_summary: str
    source_memory_ids: list[str]


class ReflectionEngine:
    """Backend-agnostic reflection: candidates → threshold filter → LLM → persist.

    Candidates arrive with pre-computed importance_score from their backend.
    Engine only filters by threshold, synthesizes, and persists.

    Args:
        candidate_provider: backend-specific provider (tabular or graph).
        writer: MemoryWriter for persisting new scene memories.
        llm_client: LLM client for synthesis calls.
        threshold: minimum importance score to trigger synthesis.
    """

    def __init__(
        self,
        candidate_provider: CandidateProvider,
        writer: MemoryWriter,
        llm_client: Any,
        threshold: float = DAILY_THRESHOLD,
        llm_retries: int = 1,
    ):
        self._provider = candidate_provider
        self._writer = writer
        self._llm = llm_client
        self._threshold = threshold
        self._llm_retries = llm_retries

    def reflect(
        self,
        user_id: str,
        *,
        since_hours: int = 24,
        existing_knowledge: str = "",
    ) -> ReflectionResult:
        """Run one reflection cycle for a user.

        1. Get candidates from backend-specific provider
        2. Score by importance, filter below threshold
        3. LLM synthesis for qualifying candidates
        4. Persist as scene-type memories (T4, conservative confidence)
        """
        import time

        start = time.time()
        result = ReflectionResult()

        # 1. Get candidates
        try:
            candidates = self._provider.get_reflection_candidates(
                user_id, since_hours=since_hours,
            )
        except Exception as e:
            logger.error("Reflection candidate retrieval failed: %s", e)
            result.errors.append(f"candidates: {e}")
            result.total_ms = (time.time() - start) * 1000
            return result

        result.candidates_found = len(candidates)
        if not candidates:
            result.total_ms = (time.time() - start) * 1000
            return result

        # 2. Score and filter
        scored = [
            (c, c.importance_score) for c in candidates
        ]
        passed = [(c, s) for c, s in scored if s >= self._threshold]
        result.candidates_passed = len(passed)

        if not passed:
            result.total_ms = (time.time() - start) * 1000
            return result

        # 3. Synthesize each qualifying candidate
        for candidate, score in passed:
            try:
                result.llm_calls += 1
                insights = self._synthesize_with_retry(candidate, existing_knowledge)

                # 4. Persist all insights from this candidate
                for insight in insights:
                    try:
                        self._persist_insight(user_id, insight)
                        result.scenes_created += 1
                    except Exception as e:
                        logger.warning("Failed to persist insight: %s", e)
                        result.errors.append(f"persist: {e}")

            except Exception as e:
                logger.warning("Reflection synthesis failed: %s", e)
                result.errors.append(f"synthesis: {e}")

        result.total_ms = (time.time() - start) * 1000
        return result

    def _synthesize_with_retry(
        self,
        candidate: ReflectionCandidate,
        existing_knowledge: str,
    ) -> list[SynthesizedInsight]:
        """Call _synthesize with retry on failure."""
        last_err: Exception | None = None
        for attempt in range(1 + self._llm_retries):
            try:
                return self._synthesize(candidate, existing_knowledge)
            except Exception as e:
                last_err = e
                if attempt < self._llm_retries:
                    logger.info("Retrying synthesis (attempt %d): %s", attempt + 1, e)
        raise last_err  # type: ignore[misc]

    def _synthesize(
        self,
        candidate: ReflectionCandidate,
        existing_knowledge: str,
    ) -> list[SynthesizedInsight]:
        """LLM synthesis for a single candidate cluster."""
        experiences = "\n\n".join(
            f"[{m.memory_type.value}] {m.content}" for m in candidate.memories
        )

        prompt = REFLECTION_SYNTHESIS_PROMPT.format(
            existing_knowledge=existing_knowledge or "(none)",
            experiences=experiences,
        )

        response = self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        raw = response if isinstance(response, str) else getattr(response, "content", str(response))
        return self._parse_insights(raw, candidate)

    def _parse_insights(
        self, raw: str, candidate: ReflectionCandidate,
    ) -> list[SynthesizedInsight]:
        """Parse LLM JSON output into SynthesizedInsight list.

        Raises ValueError on unparseable output so callers can record the error.
        """
        # Extract JSON array from response
        text = raw.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON array in LLM output: {text[:200]}")

        try:
            items = json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in LLM output: {e} — {text[:200]}") from e

        source_ids = [m.memory_id for m in candidate.memories]
        insights = []
        for item in items[:2]:  # max 2 insights per candidate
            try:
                mt = MemoryType(item["type"])
            except (KeyError, ValueError):
                continue
            conf = max(0.3, min(0.7, float(item.get("confidence", 0.5))))
            insights.append(SynthesizedInsight(
                memory_type=mt,
                content=item.get("content", ""),
                confidence=conf,
                evidence_summary=item.get("evidence_summary", ""),
                source_memory_ids=source_ids,
            ))
        return insights

    def _persist_insight(self, user_id: str, insight: SynthesizedInsight) -> None:
        """Persist a synthesized insight as a scene-type memory."""
        self._writer.store(
            user_id=user_id,
            content=insight.content,
            memory_type=insight.memory_type,
            source_event_ids=insight.source_memory_ids,
            initial_confidence=insight.confidence,
            trust_tier=TrustTier.T4_UNVERIFIED,
        )
