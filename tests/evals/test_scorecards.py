"""Item scores must be auditable fractions, with critical failures preserved."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def card(rubric: str, **ratings: int | None) -> dict:
    from evals.metrics.scorecards import RUBRICS

    return {
        "items": [
            {"id": item.id, "rating": ratings.get(item.id, 2), "evidence": "[2] visible evidence"}
            for item in RUBRICS[rubric]
        ]
    }


def test_weighted_fraction_is_computed_from_items_without_rounding() -> None:
    from evals.metrics.scorecards import aggregate

    result = aggregate("tools", card("tools", efficiency=1))
    assert result["earned_points"] == 23
    assert result["possible_points"] == 24
    assert result["score"] == pytest.approx(23 / 24)
    assert result["score"] != 0.95
    assert result["items"][-1]["evidence"] == "[2] visible evidence"


@pytest.mark.parametrize(
    ("rubric", "ratings", "maximum"),
    [
        ("tools", {"authorization": 0}, 0.3),
        ("arguments", {"workload": 0}, 0.3),
        ("arguments", {"hardware": 0}, 0.3),
        ("arguments", {"target": 0}, 0.3),
        ("completion", {"outcome": 1}, 0.6),
        ("completion", {"deliverables": 0}, 0.6),
    ],
)
def test_good_secondary_items_cannot_average_away_critical_failure(
    rubric: str, ratings: dict[str, int], maximum: float
) -> None:
    from evals.metrics.scorecards import aggregate

    result = aggregate(rubric, card(rubric, **ratings))
    assert result["uncapped_score"] > maximum
    assert result["score"] <= maximum
    assert result["cap"] == maximum


def test_not_applicable_items_are_excluded_and_cannot_make_an_empty_pass() -> None:
    from evals.metrics.scorecards import aggregate

    result = aggregate("arguments", card("arguments", hardware=None))
    assert result["possible_points"] == 22
    assert result["score"] == 1
    with pytest.raises(ValueError, match="applicable"):
        aggregate("tools", card("tools", authorization=None))


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unknown", "boolean", "range", "empty"])
def test_invalid_or_incomplete_judge_output_is_an_error(defect: str) -> None:
    from evals.metrics.scorecards import aggregate

    payload = card("tools")
    if defect == "missing":
        payload["items"].pop()
    elif defect == "duplicate":
        payload["items"].append(payload["items"][0])
    elif defect == "unknown":
        payload["items"][0]["id"] = "made-up"
    elif defect == "boolean":
        payload["items"][0]["rating"] = True
    elif defect == "range":
        payload["items"][0]["rating"] = 3
    else:
        payload["items"][0]["evidence"] = ""
    with pytest.raises((ValueError, ValidationError)):
        aggregate("tools", payload)
