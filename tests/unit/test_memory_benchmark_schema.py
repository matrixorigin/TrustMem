"""Tests for the benchmark system — schema, loader, scorer."""

import json

import pytest

from memoria.core.memory.benchmark.schema import (
    AUSSubScore,
    BenchmarkReport,
    BenchmarkScoreCard,
    MemoryAssertion,
    MQSSubScore,
    Scenario,
    ScenarioDataset,
    ScenarioStep,
    SeedMemory,
    grade_from_score,
)
from memoria.core.memory.benchmark.loader import (
    load_dataset,
    save_dataset,
    validate_dataset,
)
from memoria.core.memory.benchmark.scorer import (
    _check_assertion,
    score_dataset,
    score_scenario,
)
from memoria.core.memory.benchmark.executor import (
    AssertionResult,
    ScenarioExecution,
    StepResult,
)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def _make_scenario(**overrides) -> Scenario:
    defaults = {
        "scenario_id": "T001",
        "title": "Test scenario",
        "difficulty": "L1",
        "horizon": "short",
        "tags": ["trust"],
        "seed_memories": [SeedMemory(content="fact A")],
        "assertions": [
            MemoryAssertion(
                query="what is A?",
                expected_contents=["fact A"],
            )
        ],
    }
    defaults.update(overrides)
    return Scenario(**defaults)


def test_grade_boundaries():
    assert grade_from_score(95) == "S"
    assert grade_from_score(85) == "A"
    assert grade_from_score(75) == "B"
    assert grade_from_score(65) == "C"
    assert grade_from_score(55) == "D"


def test_scorecard_total():
    mqs = MQSSubScore(precision=90, recall=80, noise_rejection=100)
    aus = AUSSubScore(step_success_rate=100, assertion_pass_rate=80)
    card = BenchmarkScoreCard(mqs=mqs, aus=aus)
    # MQS = (90+80+100)/3 = 90, AUS = (100+80)/2 = 90
    # Total = 0.65*90 + 0.35*90 = 90
    assert card.total_score() == 90.0
    assert card.grade() == "S"


def test_scenario_requires_seed_memories():
    with pytest.raises(Exception):
        Scenario(
            scenario_id="T",
            title="T",
            difficulty="L1",
            horizon="short",
            tags=["trust"],
            seed_memories=[],
            assertions=[MemoryAssertion(query="q", expected_contents=["x"])],
        )


def test_scenario_requires_assertions():
    with pytest.raises(Exception):
        Scenario(
            scenario_id="T",
            title="T",
            difficulty="L1",
            horizon="short",
            tags=["trust"],
            seed_memories=[SeedMemory(content="x")],
            assertions=[],
        )


def test_dataset_unique_ids():
    s1 = _make_scenario(scenario_id="S1")
    s2 = _make_scenario(scenario_id="S1")
    with pytest.raises(ValueError, match="unique"):
        ScenarioDataset(dataset_id="test", version="v1.0", scenarios=[s1, s2])


def test_step_validation():
    with pytest.raises(ValueError, match="query"):
        ScenarioStep(action="retrieve")
    with pytest.raises(ValueError, match="content"):
        ScenarioStep(action="store")
    with pytest.raises(ValueError, match="content"):
        ScenarioStep(action="correct", query="find this")
    with pytest.raises(ValueError, match="query"):
        ScenarioStep(action="correct", content="new value")
    # Valid steps
    ScenarioStep(action="retrieve", query="test")
    ScenarioStep(action="store", content="test")
    ScenarioStep(action="correct", content="new", query="old", reason="fix")


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


def test_dataset_roundtrip(tmp_path):
    s = _make_scenario()
    ds = ScenarioDataset(dataset_id="test", version="v1.0", scenarios=[s])
    path = tmp_path / "ds.json"
    save_dataset(ds, path)
    loaded = load_dataset(path)
    assert loaded.dataset_id == "test"
    assert len(loaded.scenarios) == 1
    assert loaded.scenarios[0].scenario_id == "T001"


def test_validate_dataset_ok(tmp_path):
    s = _make_scenario()
    ds = ScenarioDataset(dataset_id="test", version="v1.0", scenarios=[s])
    path = tmp_path / "ds.json"
    save_dataset(ds, path)
    errors = validate_dataset(path)
    assert errors == []


def test_validate_dataset_missing_file(tmp_path):
    errors = validate_dataset(tmp_path / "nope.json")
    assert len(errors) == 1
    assert "not found" in errors[0]


# ---------------------------------------------------------------------------
# Scorer tests
# ---------------------------------------------------------------------------


def test_check_assertion_perfect():
    assertion = MemoryAssertion(
        query="what db?",
        expected_contents=["PostgreSQL"],
        excluded_contents=["MongoDB"],
    )
    result = AssertionResult(
        query="what db?",
        returned_contents=["The project uses PostgreSQL 15 with pgvector."],
    )
    detail = _check_assertion(assertion, result)
    assert detail["recall"] == 100.0
    assert detail["precision"] == 100.0
    assert detail["noise_rejection"] == 100.0
    assert detail["passed"] is True


def test_check_assertion_noise_leak():
    assertion = MemoryAssertion(
        query="what db?",
        expected_contents=["PostgreSQL"],
        excluded_contents=["MongoDB"],
    )
    result = AssertionResult(
        query="what db?",
        returned_contents=[
            "The project uses PostgreSQL 15.",
            "Previously used MongoDB but migrated.",
        ],
    )
    detail = _check_assertion(assertion, result)
    assert detail["recall"] == 100.0
    assert detail["noise_rejection"] == 0.0  # MongoDB leaked through
    assert detail["passed"] is False


def test_check_assertion_miss():
    assertion = MemoryAssertion(
        query="what db?",
        expected_contents=["PostgreSQL", "pgvector"],
        excluded_contents=[],
    )
    result = AssertionResult(
        query="what db?",
        returned_contents=["The CI pipeline runs on GitHub Actions."],
    )
    detail = _check_assertion(assertion, result)
    assert detail["recall"] == 0.0
    assert detail["precision"] == 0.0


def test_check_assertion_error():
    assertion = MemoryAssertion(query="q", expected_contents=["x"])
    result = AssertionResult(query="q", error="connection refused")
    detail = _check_assertion(assertion, result)
    assert detail["passed"] is False
    assert detail["recall"] == 0.0


def test_score_scenario_with_execution_error():
    scenario = _make_scenario()
    execution = ScenarioExecution(scenario_id="T001", error="timeout")
    result = score_scenario(scenario, execution)
    assert result.total_score == 0.0
    assert result.grade == "D"
    assert result.error == "timeout"


def test_score_scenario_perfect():
    scenario = _make_scenario()
    execution = ScenarioExecution(
        scenario_id="T001",
        assertion_results=[
            AssertionResult(query="what is A?", returned_contents=["fact A is here"])
        ],
    )
    result = score_scenario(scenario, execution)
    assert result.total_score > 80.0
    assert result.grade in ("S", "A")


def test_score_scenario_with_steps():
    scenario = _make_scenario(steps=[ScenarioStep(action="store", content="new fact")])
    execution = ScenarioExecution(
        scenario_id="T001",
        step_results=[StepResult(action="store", success=True)],
        assertion_results=[
            AssertionResult(query="what is A?", returned_contents=["fact A"])
        ],
    )
    result = score_scenario(scenario, execution)
    assert result.scorecard.aus.step_success_rate == 100.0


def test_score_scenario_failed_steps():
    scenario = _make_scenario(
        steps=[
            ScenarioStep(action="store", content="a"),
            ScenarioStep(action="store", content="b"),
        ]
    )
    execution = ScenarioExecution(
        scenario_id="T001",
        step_results=[
            StepResult(action="store", success=True),
            StepResult(action="store", success=False, error="500"),
        ],
        assertion_results=[
            AssertionResult(query="what is A?", returned_contents=["fact A"])
        ],
    )
    result = score_scenario(scenario, execution)
    assert result.scorecard.aus.step_success_rate == 50.0


def test_score_dataset_aggregation():
    s1 = _make_scenario(scenario_id="S1", difficulty="L1", tags=["trust"])
    s2 = _make_scenario(scenario_id="S2", difficulty="L2", tags=["dedup"])
    ds = ScenarioDataset(dataset_id="test", version="v1.0", scenarios=[s1, s2])

    executions = {
        "S1": ScenarioExecution(
            scenario_id="S1",
            assertion_results=[
                AssertionResult(query="what is A?", returned_contents=["fact A"])
            ],
        ),
        "S2": ScenarioExecution(
            scenario_id="S2",
            assertion_results=[
                AssertionResult(query="what is A?", returned_contents=["wrong stuff"])
            ],
        ),
    }
    report = score_dataset(ds, executions)
    assert report.scenario_count == 2
    assert "L1" in report.by_difficulty
    assert "L2" in report.by_difficulty
    assert "trust" in report.by_tag
    assert "dedup" in report.by_tag
    # S1 should score high, S2 should score low
    assert report.by_difficulty["L1"] > report.by_difficulty["L2"]


def test_score_dataset_missing_execution():
    s = _make_scenario()
    ds = ScenarioDataset(dataset_id="test", version="v1.0", scenarios=[s])
    report = score_dataset(ds, {})
    assert report.results[0].error == "no execution found"
    assert report.results[0].total_score == 0.0


def test_score_scenario_fewer_assertion_results_than_assertions():
    """Unmatched assertions must count as failures, not be silently skipped."""
    scenario = _make_scenario(
        assertions=[
            MemoryAssertion(query="q1", expected_contents=["fact A"]),
            MemoryAssertion(query="q2", expected_contents=["fact B"]),
        ]
    )
    # Executor only returned one result instead of two
    execution = ScenarioExecution(
        scenario_id="T001",
        assertion_results=[AssertionResult(query="q1", returned_contents=["fact A"])],
    )
    result = score_scenario(scenario, execution)
    # Second assertion has no result → should fail → overall score < perfect
    assert result.scorecard.aus.assertion_pass_rate < 100.0
    assert len(result.assertion_details) == 2


# ---------------------------------------------------------------------------
# Built-in dataset validation
# ---------------------------------------------------------------------------


def test_builtin_core_dataset():
    """Validate the built-in core-v1 dataset loads and parses correctly."""
    from pathlib import Path

    dataset_path = (
        Path(__file__).parent.parent.parent / "benchmarks" / "datasets" / "core-v1.json"
    )
    if not dataset_path.exists():
        pytest.skip("built-in dataset not found")
    ds = load_dataset(dataset_path)
    assert ds.dataset_id == "memoria-core-v1"
    assert len(ds.scenarios) >= 10
    # Every scenario has real data
    for s in ds.scenarios:
        assert len(s.seed_memories) >= 2, f"{s.scenario_id} needs more seed memories"
        assert len(s.assertions) >= 1, f"{s.scenario_id} needs assertions"
        for a in s.assertions:
            assert len(a.expected_contents) >= 1, (
                f"{s.scenario_id} assertion missing expected_contents"
            )


def test_report_serialization():
    report = BenchmarkReport(
        dataset_id="test",
        version="v1.0",
        scenario_count=1,
        overall_score=85.0,
        overall_grade="A",
        results=[],
    )
    data = json.loads(report.model_dump_json())
    assert data["dataset_id"] == "test"
    assert data["overall_grade"] == "A"
