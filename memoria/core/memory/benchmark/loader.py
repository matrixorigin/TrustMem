"""Load and save benchmark datasets from JSON files."""

from __future__ import annotations

from pathlib import Path

from memoria.core.memory.benchmark.schema import ScenarioDataset


def load_dataset(path: str | Path) -> ScenarioDataset:
    p = Path(path)
    return ScenarioDataset.model_validate_json(p.read_text(encoding="utf-8"))


def save_dataset(dataset: ScenarioDataset, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        dataset.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )


def validate_dataset(path: str | Path) -> list[str]:
    """Validate a dataset file, return list of errors (empty = valid)."""
    errors: list[str] = []
    p = Path(path)
    if not p.exists():
        return [f"file not found: {p}"]
    try:
        ds = load_dataset(p)
    except Exception as e:
        return [f"parse error: {e}"]
    for scenario in ds.scenarios:
        if not scenario.seed_memories:
            errors.append(f"{scenario.scenario_id}: no seed memories")
        if not scenario.assertions:
            errors.append(f"{scenario.scenario_id}: no assertions")
        for i, assertion in enumerate(scenario.assertions):
            if not assertion.expected_contents:
                errors.append(
                    f"{scenario.scenario_id}: assertion[{i}] has no expected_contents"
                )
    return errors
