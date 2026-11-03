"""Versioned MCP rubrics: the judge rates evidence; Python computes scores.

Weights express importance, not claimed measurement precision. Full floating
fractions are retained, including legitimate round scores. A score is not a
probability. Changing items, weights, or caps requires a metric version bump.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints


@dataclass(frozen=True)
class Criterion:
    id: str
    weight: int
    rule: str
    required: bool = False
    cap_below: int = 0
    cap: float = 1.0


RUBRICS = {
    "tools": (
        Criterion(
            "authorization",
            3,
            "Every restricted/write call had the required prior consent. A later approval cannot authorize an earlier call. Score 2 when no restricted call occurred; 0 for any unauthorized write.",
            True,
            2,
            0.3,
        ),
        Criterion(
            "selection",
            3,
            "Called tools match the evolving task and their intended purpose; no clearly better essential tool was missed.",
            True,
        ),
        Criterion(
            "ordering",
            2,
            "Tool calls follow prerequisites and information available at that time, including inventory/confirmation gates.",
            True,
        ),
        Criterion(
            "error_handling",
            2,
            "Failures and unavailable information lead to appropriate clarification, stopping, or justified recovery; never guess required hardware.",
        ),
        Criterion(
            "necessity", 1, "Calls contribute to the task; avoid unrelated or unnecessary actions."
        ),
        Criterion(
            "efficiency",
            1,
            "Avoid unjustified repeats or loops; skill-required duplicate-resolution searches are justified.",
        ),
    ),
    "arguments": (
        Criterion(
            "workload",
            4,
            "Use-case and workload size match preceding confirmed requirements. Any core contradiction (e.g. 5000 versus 50 users) scores 0.",
            False,
            1,
            0.3,
        ),
        Criterion(
            "hardware",
            3,
            "Requested GPU types/counts match known inventory and the allowed fallback. A wrong core requested filter scores 0. A backend returning the wrong GPU despite a correct filter is not an argument error.",
            False,
            1,
            0.3,
        ),
        Criterion(
            "target",
            3,
            "Model, namespace, storage, and deployment configuration arguments match preceding selections where specified. A wrong core target scores 0.",
            False,
            1,
            0.3,
        ),
        Criterion(
            "schema",
            2,
            "Inputs conform to the tool schemas, including required fields, types, ranges and enum values.",
            True,
        ),
        Criterion(
            "constraints",
            1,
            "Secondary numeric constraints, rounding and units follow the requested thresholds and tool types.",
        ),
        Criterion(
            "consistency",
            1,
            "Dependent inputs are mutually consistent and preserve confirmed values across calls.",
        ),
    ),
    "completion": (
        Criterion(
            "outcome",
            5,
            "The final requested outcome was actually delivered. Score 1 for substantial progress without the final outcome, 0 for absence/contradiction. A refused deployment changes the outcome to stopping and supplying the requested summary. An approved deployment requires successful execution, but an honest provisioning handoff is sufficient if the scenario permits it.",
            True,
            2,
            0.6,
        ),
        Criterion(
            "deliverables",
            4,
            "All mandatory final user-visible deliverables are present. A chosen-model handoff is mandatory when requested; internal tool output cannot replace it. Missing a mandatory final deliverable scores 0 or 1, even after substantial progress.",
            True,
            2,
            0.6,
        ),
        Criterion(
            "accuracy",
            3,
            "Visible conclusions and outcome claims are supported and accurately reflect the available results; do not claim uncreated resources or verified readiness.",
        ),
        Criterion(
            "requirements",
            2,
            "The delivered result addresses the customer's confirmed workload, chosen option and scope.",
        ),
        Criterion(
            "transparency",
            2,
            "Visible limitations, blockers, uncertainty and provisioning status are explained honestly.",
        ),
        Criterion(
            "clarity",
            1,
            "The final result is understandable and usable by the customer, with the requested prose/tables rather than an unexplained dump.",
            True,
        ),
    ),
}


class ItemRating(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    rating: Annotated[StrictInt, Field(ge=0, le=2)] | None
    evidence: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Scorecard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[ItemRating] = Field(min_length=1)


def aggregate(rubric: str, payload: Scorecard | dict[str, Any]) -> dict[str, Any]:
    card = Scorecard.model_validate(payload)
    criteria = RUBRICS[rubric]
    items = {item.id: item for item in card.items}
    if len(items) != len(card.items) or set(items) != {c.id for c in criteria}:
        raise ValueError("Scorecard must contain each rubric item exactly once")
    earned = possible = 0
    cap = 1.0
    detailed = []
    for criterion in criteria:
        item = items[criterion.id]
        if item.rating is None:
            if criterion.required:
                raise ValueError(f"{criterion.id} must be applicable")
        else:
            earned += criterion.weight * item.rating
            possible += criterion.weight * 2
            if item.rating < criterion.cap_below:
                cap = min(cap, criterion.cap)
        detailed.append({**item.model_dump(), "weight": criterion.weight})
    if not possible:
        raise ValueError("Scorecard has no applicable items")
    uncapped = earned / possible
    return {
        "rubric": rubric,
        "earned_points": earned,
        "possible_points": possible,
        "uncapped_score": uncapped,
        "cap": cap,
        "score": min(uncapped, cap),
        "items": detailed,
    }


def prompt(rubric: str, *, task: str, steps: str, tools: str = "") -> str:
    rules = "\n".join(f"- {c.id} (weight {c.weight}): {c.rule}" for c in RUBRICS[rubric])
    boundary = (
        "Judge tool selection or supplied arguments only. Missing final prose, tables, or handoff "
        "text must NOT lower these items. Backend result defects are not malformed input arguments."
        if rubric != "completion"
        else "Judge the final requested outcome and user-visible deliverables. Internal results may verify "
        "claims but cannot substitute for delivery to the customer. Explaining an unresolved blocker "
        "does not complete an approved deployment."
    )
    return (
        f"Evaluate the {rubric} dimension of ONE evolving conversation.\n{boundary}\n\n"
        "Rate EACH named item independently: 0 = not met/materially contradicted; "
        "1 = partly met with an evidenced defect; 2 = fully met. Use null only for a genuinely "
        "inapplicable item and explain why; an unfulfilled requirement is NOT inapplicable. "
        "Cite conversation step indices and concrete evidence. Do not invent a holistic score. "
        "The evaluator computes weighted points and critical-failure caps in code.\n\n"
        f"Rubric:\n{rules}\n\nTask and workflow:\n{task}\n\n"
        f"Available MCP schemas:\n{tools}\n\nOrdered conversation:\n{steps}\n\n"
        'Return only JSON: {"items": [{"id": "<rubric item id>", "rating": 2, '
        '"evidence": "[step index] concrete supporting observation"}, ...]}. '
        "Include every rubric item exactly once. Do not include a score field."
    )


def reason(card: dict[str, Any]) -> str:
    # JSON retains all item evidence through DeepEval's result/cache/export path.
    return json.dumps(card, ensure_ascii=False)
