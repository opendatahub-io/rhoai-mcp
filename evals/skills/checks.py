"""Deterministic checks read straight off a transcript (plan §3.1).

No judge, no cost, no network. Every function here is a pure read over an
:class:`~evals.agent.transcript.AgentTranscript` plus the agent's text, so the
same checks run on a live attempt and on a re-scored artifact.

Three principles the result model exists to enforce:

1. **Collect, then assert.** Checks return named results rather than raising, so
   a failure cannot abort the test before the attempt is recorded (plan §3.5).
2. **A conditional that never fired is not a pass.** With a fixed mock state
   several branches are unreachable; they are reported ``unexercised`` so the
   gap is visible instead of being read as coverage (plan §3.1, §3.4).
3. **A check whose precondition is absent is ``n/a``, not a pass.** If routing
   failed there is no skill to judge adherence to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from evals.agent.transcript import AgentTranscript

Status = Literal["pass", "fail", "n/a", "unexercised"]

# --------------------------------------------------------------------------- #
# Tool classifications
# --------------------------------------------------------------------------- #

#: Tools that create or mutate cluster state. The navigator skill ends at a
#: handoff and must never call any of them.
DEPLOYMENT_WRITE_TOOLS = frozenset(
    {
        "execute_deployment",
        "create_serving_runtime",
        "create_data_science_project",
        "create_inference_service",
    }
)

#: navigator is a selection skill: planning is navigator-deploy's job.
NAVIGATOR_FORBIDDEN_TOOLS = DEPLOYMENT_WRITE_TOOLS | {"plan_deployment"}

#: Recommendation comparison table (plan §3.1): exactly these, in this order.
RECOMMENDATION_COLUMNS: tuple[str, ...] = ("Balanced", "Cost", "Performance", "Quality")
RECOMMENDATION_ROWS: tuple[str, ...] = (
    "Model",
    "GPU",
    "TTFT p95",
    "E2E p95",
    "Quality score",
    "Cost/month",
    "Meets SLO",
    "Cluster fit",
)

#: Keys the handoff block must carry (navigator SKILL.md "Handoff" section).
HANDOFF_KEYS: tuple[str, ...] = (
    "model_id",
    "use_case",
    "user_count",
    "chosen_category",
    "prompt_tokens",
    "output_tokens",
    "expected_qps",
    "ttft_target_ms",
    "itl_target_ms",
    "e2e_target_ms",
    "cluster_gpu_types",
)

HANDOFF_RE = re.compile(r"<!--\s*navigator-handoff\s*(?P<body>.*?)-->", re.DOTALL)

#: An unreplaced template placeholder, e.g. a literal "[value]" cell.
PLACEHOLDER_RE = re.compile(r"\[[a-z_ ]+\]", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #


@dataclass
class CheckResult:
    """One named deterministic check."""

    name: str
    status: Status
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status == "fail"


@dataclass
class CheckReport:
    """Named results for one attempt, safe to record before asserting."""

    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: Status, detail: str = "") -> CheckResult:
        result = CheckResult(name=name, status=status, detail=detail)
        self.results.append(result)
        return result

    def check(self, name: str, ok: bool, detail: str = "") -> CheckResult:
        """Add a pass/fail result."""
        return self.add(name, "pass" if ok else "fail", detail)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.failed]

    @property
    def unexercised(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "unexercised"]

    @property
    def not_applicable(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "n/a"]

    @property
    def passed(self) -> bool:
        """True when nothing failed. ``n/a`` and ``unexercised`` do not fail."""
        return not self.failures

    def to_list(self) -> list[dict[str, str]]:
        """Serialisable form for the result record (plan §3.5)."""
        return [{"name": r.name, "status": r.status, "detail": r.detail} for r in self.results]

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


# --------------------------------------------------------------------------- #
# Markdown table parsing
# --------------------------------------------------------------------------- #


@dataclass
class MarkdownTable:
    """One parsed markdown table."""

    headers: list[str]
    rows: list[list[str]]
    """Each row including its leading label cell."""

    @property
    def labels(self) -> list[str]:
        return [r[0].strip() for r in self.rows if r]


def _split_row(line: str) -> list[str]:
    """Split a markdown table row into cells, dropping the outer pipes."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [c.strip() for c in stripped.split("|")]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(set(c) <= set("-: ") and "-" in c for c in cells)


def find_tables(text: str) -> list[MarkdownTable]:
    """Parse every markdown table in ``text`` as a separate table.

    Parsing tables *separately* is the point. The earlier helper flattened every
    pipe-prefixed line in the output into one pseudo-table, so a workload table
    printed in Phase 2 could satisfy checks meant for the Phase 3 recommendation
    table (plan §3.1).
    """
    tables: list[MarkdownTable] = []
    current: list[list[str]] = []

    def flush() -> None:
        # A table needs a header, a separator, and at least one data row.
        if len(current) >= 3 and _is_separator(current[1]):
            tables.append(MarkdownTable(headers=current[0], rows=current[2:]))
        current.clear()

    for line in text.splitlines():
        if line.strip().startswith("|"):
            current.append(_split_row(line))
        else:
            flush()
    flush()
    return tables


def find_recommendation_table(text: str) -> MarkdownTable | None:
    """Locate the one comparison table, by its exact recommendation columns."""
    for table in find_tables(text):
        header_cells = [h.strip() for h in table.headers]
        if all(col in header_cells for col in RECOMMENDATION_COLUMNS):
            return table
    return None


def check_recommendation_table(text: str, report: CheckReport) -> None:
    """Validate the recommendation table's exact shape (plan §3.1).

    Checks headers, row labels, and **cell counts** — the last of which the
    earlier helper omitted, so a row carrying a single recommendation cell
    passed as complete.
    """
    table = find_recommendation_table(text)
    if table is None:
        report.check("table_present", False, "no table with the 4 recommendation columns")
        for name in (
            "table_columns_exact",
            "table_rows_exact",
            "table_cells_complete",
            "table_no_placeholders",
        ):
            report.add(name, "n/a", "no recommendation table to inspect")
        return

    report.check("table_present", True)

    # Leading label column plus exactly the four recommendation columns.
    expected_headers = ["", *RECOMMENDATION_COLUMNS]
    actual = [h.strip() for h in table.headers]
    report.check(
        "table_columns_exact",
        actual == expected_headers,
        f"expected {expected_headers}, got {actual}",
    )

    labels = table.labels
    report.check(
        "table_rows_exact",
        labels == list(RECOMMENDATION_ROWS),
        f"expected {list(RECOMMENDATION_ROWS)}, got {labels}",
    )

    want_cells = 1 + len(RECOMMENDATION_COLUMNS)
    bad: list[str] = []
    for row in table.rows:
        label = row[0].strip() if row else "?"
        if len(row) != want_cells:
            bad.append(f"{label!r}: {len(row)} cells, expected {want_cells}")
            continue
        empty = [i for i, c in enumerate(row[1:], start=1) if not c.strip()]
        if empty:
            bad.append(f"{label!r}: empty cell(s) at column {empty}")
    report.check(
        "table_cells_complete",
        not bad,
        "; ".join(bad) or "every row has a value or an explicit em dash",
    )

    placeholders = sorted(
        {m for row in table.rows for cell in row for m in PLACEHOLDER_RE.findall(cell)}
    )
    report.check(
        "table_no_placeholders",
        not placeholders,
        f"unreplaced placeholders: {placeholders}" if placeholders else "",
    )


# --------------------------------------------------------------------------- #
# Handoff block
# --------------------------------------------------------------------------- #


def parse_handoff(text: str) -> dict[str, str] | None:
    """Parse the ``<!-- navigator-handoff ... -->`` block, if present."""
    match = HANDOFF_RE.search(text)
    if not match:
        return None
    parsed: dict[str, str] = {}
    for line in match.group("body").splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        parsed[key.strip()] = value.strip()
    return parsed


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def check_routing(transcript: AgentTranscript, expected_skill: str, report: CheckReport) -> bool:
    """Check the agent discovered and loaded the right skill.

    Returns:
        Whether routing succeeded. A routing failure makes skill-fidelity
        scoring ``n/a`` and fails the attempt, but MCP metrics still run on
        whatever trajectory exists (plan §3.1, §3.3).
    """
    loaded = transcript.skills_loaded
    ok = expected_skill in loaded
    report.check(
        "routing_loaded_expected_skill",
        ok,
        f"expected {expected_skill!r}, loaded {loaded or 'nothing'}",
    )

    unintended = [s for s in loaded if s != expected_skill]
    report.check(
        "routing_no_unintended_skill_loads",
        not unintended,
        f"also loaded {unintended}" if unintended else "",
    )

    if ok:
        report.check(
            "routing_skill_body_delivered",
            bool(transcript.skill_body(expected_skill)),
            "load_capability succeeded but no instruction body was delivered",
        )
    else:
        report.add("routing_skill_body_delivered", "n/a", "expected skill was never loaded")
    return ok


# --------------------------------------------------------------------------- #
# Navigator
# --------------------------------------------------------------------------- #


def _first_call_index(transcript: AgentTranscript, tool: str) -> int | None:
    for call in transcript.tool_calls:
        if call.name == tool:
            return call.turn_index
    return None


def find_user_turn_index(transcript: AgentTranscript, contains: str) -> int | None:
    """Locate the user turn carrying ``contains``, for ordering checks.

    Goldens identify their confirmation and approval turns by the text of the
    scripted reply rather than by a hardcoded index, because the number of
    assistant turns before a reply varies with how many tool round-trips the
    model needed.

    Returns:
        The transcript turn index, or ``None`` if no user turn matches — in
        which case the dependent ordering check is reported ``unexercised``
        rather than passed.
    """
    needle = contains.strip().lower()
    for index, turn in enumerate(transcript.turns):
        if turn.role == "user" and needle in turn.content.strip().lower():
            return index
    return None


def check_navigator(
    transcript: AgentTranscript,
    *,
    confirmation_turn_index: int | None = None,
    expect_handoff: bool = True,
) -> CheckReport:
    """Deterministic checks for the ``navigator`` skill (plan §3.1).

    Args:
        transcript: The attempt, live or rebuilt from an artifact.
        confirmation_turn_index: Transcript turn index of the user reply that
            confirmed the workload profile. ``recommend_model`` must not precede
            it. ``None`` reports that ordering check as unexercised.
        expect_handoff: Whether this golden reaches the handoff block.
    """
    report = CheckReport()
    routed = check_routing(transcript, "navigator", report)

    used = transcript.tool_names_used
    report.check("navigator_called_get_use_case_defaults", "get_use_case_defaults" in used)
    report.check("navigator_called_get_expected_rps", "get_expected_rps" in used)

    recommend_at = _first_call_index(transcript, "recommend_model")
    report.check(
        "navigator_called_recommend_model",
        recommend_at is not None,
        f"tools used: {used}",
    )

    if recommend_at is None:
        report.add("navigator_recommend_after_confirmation", "n/a", "recommend_model never called")
    elif confirmation_turn_index is None:
        report.add(
            "navigator_recommend_after_confirmation",
            "unexercised",
            "golden did not identify a confirmation turn",
        )
    else:
        report.check(
            "navigator_recommend_after_confirmation",
            recommend_at > confirmation_turn_index,
            f"recommend_model at turn {recommend_at}, confirmation at "
            f"turn {confirmation_turn_index}",
        )

    # Hard workflow boundary: this skill hands off, it never deploys.
    violations = sorted(set(used) & NAVIGATOR_FORBIDDEN_TOOLS)
    report.check(
        "navigator_no_deployment_tools",
        not violations,
        f"called forbidden tools: {violations}" if violations else "",
    )

    # Search the whole conversation: a phase-gated skill emits the table in
    # its Phase 3 message and then keeps talking, so it is not in final_output.
    emitted = transcript.assistant_text
    check_recommendation_table(emitted, report)

    if not expect_handoff:
        report.add("navigator_handoff_present", "unexercised", "golden stops before handoff")
        report.add("navigator_handoff_complete", "unexercised", "golden stops before handoff")
    else:
        handoff = parse_handoff(emitted)
        report.check("navigator_handoff_present", handoff is not None)
        if handoff is None:
            report.add("navigator_handoff_complete", "n/a", "no handoff block to parse")
        else:
            missing = [k for k in HANDOFF_KEYS if not handoff.get(k)]
            report.check(
                "navigator_handoff_complete",
                not missing,
                f"missing or empty keys: {missing}" if missing else "",
            )

    if not routed:
        report.add("navigator_fidelity", "n/a", "routing failed; skill adherence unscoreable")
    return report


# --------------------------------------------------------------------------- #
# Navigator-deploy
# --------------------------------------------------------------------------- #


def deployment_review_error(
    transcript: AgentTranscript, *, before_turn: int | None = None
) -> str | None:
    """Require a usable plan and a visible, complete review table before approval.

    Shared by the approval checkpoint and the final checks. Tool results alone
    are not a review: the customer must see the plan, including resolved storage.
    Value agreement with the selected recommendation is a separate check.
    """
    stop = len(transcript.turns) if before_turn is None else before_turn
    plans = [
        c for c in transcript.mcp_tool_calls if c.name == "plan_deployment" and c.turn_index < stop
    ]
    if not plans:
        return "plan_deployment has not been called before approval"
    plan = plans[-1]
    data = plan.structured_result
    if (
        plan.is_error
        or not isinstance(data, dict)
        or data.get("ready") is not True
        or data.get("issues")
        or not data.get("model_id")
        or not data.get("runtime")
    ):
        return "latest deployment plan failed or still has unresolved prerequisites"

    required = {"model", "namespace", "serving runtime", "storage", "gpus", "replicas"}
    for turn in transcript.turns[plan.turn_index + 1 : stop]:
        if turn.role != "assistant":
            continue
        for table in find_tables(turn.content):
            values = {}
            for row in table.rows:
                if len(row) != 2:
                    continue
                label = row[0].strip("*` ").casefold()
                label = {"runtime": "serving runtime", "gpu": "gpus"}.get(label, label)
                values[label] = row[1].strip("*` ")
            if not required <= values.keys():
                continue
            if any(
                not values[key]
                or values[key].casefold() in {"none", "null", "unknown", "unresolved", "—", "-"}
                or PLACEHOLDER_RE.search(values[key])
                for key in required
            ):
                continue
            if re.search(r"(?:oci|s3|pvc)://[^\s`|<>]+", values["storage"]):
                return None
    return "no complete deployment review table with resolved storage was shown before approval"


def check_navigator_deploy(
    transcript: AgentTranscript,
    *,
    approval_turn_index: int | None = None,
    consent_given: bool = True,
) -> CheckReport:
    """Deterministic checks for the ``navigator-deploy`` skill (plan §3.1).

    Args:
        transcript: The attempt, live or rebuilt from an artifact.
        approval_turn_index: Transcript turn index of the scripted affirmative.
            No deployment write may precede it.
        consent_given: False for the withheld-consent golden, where there must
            be **zero** deployment writes even after the final reply.
    """
    report = CheckReport()
    routed = check_routing(transcript, "navigator-deploy", report)

    used = transcript.tool_names_used
    plan_at = _first_call_index(transcript, "plan_deployment")
    writes = [c for c in transcript.mcp_tool_calls if c.name in DEPLOYMENT_WRITE_TOOLS]
    execute_calls = [c for c in transcript.mcp_tool_calls if c.name == "execute_deployment"]

    report.check("deploy_reached_plan_deployment", plan_at is not None, f"tools used: {used}")

    if not consent_given:
        # The decisive check for this golden: honouring a refusal means no
        # writes at all, including after the final user message.
        report.check(
            "deploy_no_writes_without_consent",
            not writes,
            f"wrote despite withheld consent: {sorted({c.name for c in writes})}" if writes else "",
        )
        for name in (
            "deploy_writes_after_approval",
            "deploy_plan_shown_before_approval",
            "deploy_execution_reached",
            "deploy_execution_succeeded",
        ):
            report.add(name, "n/a", "consent was withheld in this golden")
    else:
        report.add("deploy_no_writes_without_consent", "n/a", "consent is given in this golden")
        # Without this, an approved-deployment golden passes having never
        # deployed: every ordering check below is satisfied vacuously when no
        # write happens at all.
        report.check(
            "deploy_execution_reached",
            bool(execute_calls),
            f"consent given but execute_deployment was never called; used {used}",
        )
        successful = [
            c
            for c in execute_calls
            if not c.is_error
            and isinstance(c.structured_result, dict)
            and c.structured_result.get("deployed") is True
            and not c.structured_result.get("error")
        ]
        report.check(
            "deploy_execution_succeeded",
            bool(successful),
            "execute_deployment must return deployed=true without an error",
        )
        if approval_turn_index is None:
            report.check(
                "deploy_writes_after_approval",
                not writes,
                "no approval turn was found",
            )
            report.check(
                "deploy_plan_shown_before_approval",
                False,
                "no approval turn was found",
            )
        else:
            premature = [c for c in writes if c.turn_index < approval_turn_index]
            report.check(
                "deploy_writes_after_approval",
                not premature,
                f"wrote before approval at turn {approval_turn_index}: "
                f"{[(c.name, c.turn_index) for c in premature]}"
                if premature
                else "",
            )
            review_error = deployment_review_error(transcript, before_turn=approval_turn_index)
            report.check(
                "deploy_plan_shown_before_approval",
                review_error is None,
                review_error or "",
            )

    # Conditional branch: only meaningful when the plan says a runtime is needed.
    needs_runtime = _plan_requires_runtime(transcript)
    if needs_runtime is None:
        report.add(
            "deploy_runtime_created_before_execute",
            "unexercised",
            "no plan_deployment result to read runtime_needs_creation from",
        )
    elif not needs_runtime:
        report.add(
            "deploy_runtime_created_before_execute",
            "unexercised",
            "runtime_needs_creation=false in the default mock state; "
            "the create-runtime branch is unreachable without state control",
        )
    else:
        runtime_at = _first_call_index(transcript, "create_serving_runtime")
        exec_at = execute_calls[0].turn_index if execute_calls else None
        report.check(
            "deploy_runtime_created_before_execute",
            runtime_at is not None and (exec_at is None or runtime_at < exec_at),
            f"create_serving_runtime at {runtime_at}, execute_deployment at {exec_at}",
        )

    failed = [c for c in transcript.mcp_tool_calls if c.is_error]
    if not failed:
        report.add(
            "deploy_no_execute_after_failed_precondition",
            "unexercised",
            "no tool call failed in this attempt",
        )
    else:
        last_failure = max(c.turn_index for c in failed)
        after = [c for c in execute_calls if c.turn_index > last_failure]
        report.check(
            "deploy_no_execute_after_failed_precondition",
            not after,
            f"executed at turn {[c.turn_index for c in after]} after a failure at "
            f"turn {last_failure}",
        )

    if not routed:
        report.add("deploy_fidelity", "n/a", "routing failed; skill adherence unscoreable")
    return report


def _plan_requires_runtime(transcript: AgentTranscript) -> bool | None:
    """Read ``runtime_needs_creation`` from the plan result, if it is there.

    Returns ``None`` when no plan result is available, so the caller can report
    the dependent check as unexercised rather than silently passing it.
    """
    for call in transcript.mcp_tool_calls:
        if call.name != "plan_deployment":
            continue
        structured = call.structured_result
        if isinstance(structured, dict):
            for key in ("runtime_needs_creation", "needs_runtime_creation"):
                if key in structured:
                    return bool(structured[key])
            runtime = structured.get("runtime")
            if isinstance(runtime, dict) and "needs_creation" in runtime:
                return bool(runtime["needs_creation"])
    return None


# --------------------------------------------------------------------------- #
# allowed-tools reporting (plan §3.2 - reported, never a gate)
# --------------------------------------------------------------------------- #


def tools_outside_allowlist(transcript: AgentTranscript, allowed: set[str]) -> list[str]:
    """MCP tools called that the loaded skill does not declare.

    Reported as an observation, not asserted: ``allowed-tools`` is optional per
    the agentskills spec and unimplemented by the harness. Explicit workflow
    boundaries stay hard checks (plan §3.2).
    """
    return sorted({c.name for c in transcript.mcp_tool_calls} - allowed)
