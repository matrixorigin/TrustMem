"""Score benchmark executions against ground truth.

The scorer compares actual retrieval results against the assertions defined
in each scenario. It does NOT rely on self-reported metrics from the executor.
"""

from __future__ import annotations

from memoria.core.memory.benchmark.executor import (
    AssertionResult,
    ScenarioExecution,
)
from memoria.core.memory.benchmark.schema import (
    AUSSubScore,
    BenchmarkReport,
    BenchmarkScoreCard,
    ChallengeTag,
    DifficultyLevel,
    HorizonLevel,
    MemoryAssertion,
    MQSSubScore,
    Scenario,
    ScenarioDataset,
    ScenarioResult,
    grade_from_score,
)


def _score_contents(assertion: MemoryAssertion, returned_set: list[str]) -> dict:
    """Score a set of returned contents against assertion ground truth."""
    hits = sum(
        1
        for expected in assertion.expected_contents
        if any(expected.lower() in c.lower() for c in returned_set)
    )
    recall = (
        100.0 * hits / len(assertion.expected_contents)
        if assertion.expected_contents
        else 100.0
    )

    if returned_set:
        relevant_count = sum(
            1
            for c in returned_set
            if any(exp.lower() in c.lower() for exp in assertion.expected_contents)
        )
        precision = 100.0 * relevant_count / len(returned_set)
    else:
        precision = 0.0

    if assertion.excluded_contents:
        noise_hits = sum(
            1
            for excluded in assertion.excluded_contents
            if any(excluded.lower() in c.lower() for c in returned_set)
        )
        noise_rejection = (
            100.0
            * (len(assertion.excluded_contents) - noise_hits)
            / len(assertion.excluded_contents)
        )
    else:
        noise_rejection = 100.0

    passed = recall >= 80.0 and noise_rejection >= 80.0
    return {
        "passed": passed,
        "recall": round(recall, 2),
        "precision": round(precision, 2),
        "noise_rejection": round(noise_rejection, 2),
        "expected_count": len(assertion.expected_contents),
        "expected_found": hits,
        "excluded_count": len(assertion.excluded_contents),
        "returned_count": len(returned_set),
    }


def _check_assertion(assertion: MemoryAssertion, result: AssertionResult) -> dict:
    """Check one assertion against its execution result. Returns detail dict.

    If follow-up strategies were executed, each is scored independently.
    The assertion passes if the base query OR any follow-up strategy passes.
    The best-scoring variant is used for the final metrics.
    """
    if result.error:
        return {
            "query": assertion.query,
            "passed": False,
            "error": result.error,
            "precision": 0.0,
            "recall": 0.0,
            "noise_rejection": 100.0,
        }

    # Score base query
    base = _score_contents(assertion, result.returned_contents)
    base["query"] = assertion.query
    base["variant"] = "base"

    # Score each follow-up strategy
    follow_up_variants = []
    for fur in result.follow_up_results:
        v = _score_contents(assertion, fur.returned_contents)
        v["variant"] = fur.strategy_name
        v["rounds"] = fur.rounds
        v["queries_used"] = fur.queries_used
        follow_up_variants.append(v)

    # Use base query score as the primary result; attach follow-up variants for comparison.
    # The assertion passes if base OR any follow-up variant passes.
    all_variants = [base] + follow_up_variants
    base["passed"] = any(v["passed"] for v in all_variants)
    if follow_up_variants:
        base["follow_up_variants"] = follow_up_variants
    return base


def score_scenario(scenario: Scenario, execution: ScenarioExecution) -> ScenarioResult:
    """Score a single scenario execution against its ground truth."""
    if execution.error:
        empty_mqs = MQSSubScore(precision=0, recall=0, noise_rejection=100)
        empty_aus = AUSSubScore(step_success_rate=0, assertion_pass_rate=0)
        card = BenchmarkScoreCard(mqs=empty_mqs, aus=empty_aus)
        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            title=scenario.title,
            difficulty=scenario.difficulty,
            horizon=scenario.horizon,
            tags=scenario.tags,
            scorecard=card,
            total_score=0.0,
            grade="D",
            error=execution.error,
        )

    # Score assertions — pad missing results so every assertion is evaluated
    assertion_details = []
    for i, assertion in enumerate(scenario.assertions):
        if i < len(execution.assertion_results):
            result = execution.assertion_results[i]
        else:
            result = AssertionResult(
                query=assertion.query, error="no result returned by executor"
            )
        detail = _check_assertion(assertion, result)
        assertion_details.append(detail)

    # Aggregate MQS from assertion checks
    if assertion_details:
        avg_precision = sum(d["precision"] for d in assertion_details) / len(
            assertion_details
        )
        avg_recall = sum(d["recall"] for d in assertion_details) / len(
            assertion_details
        )
        avg_noise = sum(d["noise_rejection"] for d in assertion_details) / len(
            assertion_details
        )
        assertion_pass_rate = (
            100.0
            * sum(1 for d in assertion_details if d["passed"])
            / len(assertion_details)
        )
    else:
        avg_precision = avg_recall = avg_noise = 0.0
        assertion_pass_rate = 0.0

    # Step success rate
    step_details = [
        {"action": r.action, "success": r.success, "error": r.error}
        for r in execution.step_results
    ]
    if execution.step_results:
        step_success_rate = (
            100.0
            * sum(1 for r in execution.step_results if r.success)
            / len(execution.step_results)
        )
    else:
        step_success_rate = 100.0  # no steps = nothing to fail

    mqs = MQSSubScore(
        precision=round(avg_precision, 2),
        recall=round(avg_recall, 2),
        noise_rejection=round(avg_noise, 2),
    )
    aus = AUSSubScore(
        step_success_rate=round(step_success_rate, 2),
        assertion_pass_rate=round(assertion_pass_rate, 2),
    )
    card = BenchmarkScoreCard(mqs=mqs, aus=aus)
    total = card.total_score()

    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        title=scenario.title,
        difficulty=scenario.difficulty,
        horizon=scenario.horizon,
        tags=scenario.tags,
        scorecard=card,
        total_score=round(total, 2),
        grade=grade_from_score(total),
        assertion_details=assertion_details,
        step_details=step_details,
    )


def score_dataset(
    dataset: ScenarioDataset,
    executions: dict[str, ScenarioExecution],
) -> BenchmarkReport:
    """Score all scenarios in a dataset."""
    results: list[ScenarioResult] = []
    scores_by_difficulty: dict[DifficultyLevel, list[float]] = {}
    scores_by_horizon: dict[HorizonLevel, list[float]] = {}
    scores_by_tag: dict[ChallengeTag, list[float]] = {}

    for scenario in dataset.scenarios:
        execution = executions.get(scenario.scenario_id)
        if execution is None:
            execution = ScenarioExecution(
                scenario_id=scenario.scenario_id,
                error="no execution found",
            )
        result = score_scenario(scenario, execution)
        results.append(result)

        scores_by_difficulty.setdefault(scenario.difficulty, []).append(
            result.total_score
        )
        scores_by_horizon.setdefault(scenario.horizon, []).append(result.total_score)
        for tag in scenario.tags:
            scores_by_tag.setdefault(tag, []).append(result.total_score)

    def _avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    all_scores = [r.total_score for r in results]
    overall = _avg(all_scores)

    return BenchmarkReport(
        dataset_id=dataset.dataset_id,
        version=dataset.version,
        scenario_count=len(results),
        overall_score=round(overall, 2),
        overall_grade=grade_from_score(overall),
        by_difficulty={k: round(_avg(v), 2) for k, v in scores_by_difficulty.items()},
        by_horizon={k: round(_avg(v), 2) for k, v in scores_by_horizon.items()},
        by_tag={k: round(_avg(v), 2) for k, v in scores_by_tag.items()},
        results=results,
    )
