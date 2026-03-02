"""HTTP client for Lightspeed Core Stack (LCS) REST API.

Sends evaluation queries to LCS and parses responses into structured
results compatible with DeepEval scoring helpers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LCSToolCall:
    """A single tool call made during an LCS query."""

    name: str
    arguments: dict[str, Any]
    result: str
    call_id: str = ""


@dataclass
class LCSResult:
    """Result of an LCS query, analogous to the old AgentResult."""

    task: str
    final_output: str
    tool_calls: list[LCSToolCall] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    conversation_id: str = ""

    @property
    def tool_names_used(self) -> list[str]:
        """Get the list of tool names called, in order."""
        return [tc.name for tc in self.tool_calls]


class LCSClient:
    """HTTP client for the Lightspeed Core Stack REST API.

    Sends queries to LCS's /v1/query endpoint and parses the response
    into an LCSResult for downstream DeepEval scoring.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8443",
        timeout: int = 300,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def health_check(self) -> bool:
        """Check if LCS is ready to accept queries."""
        try:
            response = await self._client.get("/healthz")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def query(self, task: str) -> LCSResult:
        """Send a query to LCS and parse the response.

        Args:
            task: Natural language task for the agent.

        Returns:
            LCSResult with final output, tool calls, and messages.
        """
        payload = {"query": task}
        response = await self._client.post("/v1/query", json=payload)
        response.raise_for_status()
        data = response.json()

        return self._parse_response(task, data)

    def _parse_response(self, task: str, data: dict[str, Any]) -> LCSResult:
        """Parse an LCS query response into an LCSResult.

        Handles the LCS response format:
        - response: final text answer
        - tool_calls: list of {id, name, args}
        - tool_results: list of {id, status, content, round}
        - conversation_id: conversation tracking
        """
        final_output = data.get("response", "")
        conversation_id = data.get("conversation_id", "")

        # Parse tool calls and match with results
        raw_tool_calls = data.get("tool_calls", [])
        raw_tool_results = data.get("tool_results", [])

        # Index results by tool call ID for matching
        results_by_id: dict[str, dict[str, Any]] = {}
        for tr in raw_tool_results:
            tr_id = tr.get("id", "")
            if tr_id:
                results_by_id[tr_id] = tr

        tool_calls: list[LCSToolCall] = []
        for tc in raw_tool_calls:
            tc_id = tc.get("id", "")
            tc_name = tc.get("name", "")
            tc_args = tc.get("args", {})
            if isinstance(tc_args, str):
                import json

                try:
                    tc_args = json.loads(tc_args)
                except (json.JSONDecodeError, TypeError):
                    tc_args = {"raw": tc_args}

            # Match with result
            matched_result = results_by_id.get(tc_id, {})
            result_content = matched_result.get("content", "")
            if isinstance(result_content, dict | list):
                import json

                result_content = json.dumps(result_content, default=str)

            tool_calls.append(
                LCSToolCall(
                    name=tc_name,
                    arguments=tc_args,
                    result=str(result_content),
                    call_id=tc_id,
                )
            )

        # Count turns from tool result rounds
        turns = 0
        if raw_tool_results:
            rounds = {tr.get("round", 0) for tr in raw_tool_results}
            turns = max(rounds) if rounds else 0
        # At minimum 1 turn if we got a response
        if final_output:
            turns = max(turns, 1)

        # Build OpenAI-format messages for DeepEval
        messages = self._build_messages(task, final_output, tool_calls, raw_tool_results)

        return LCSResult(
            task=task,
            final_output=final_output,
            tool_calls=tool_calls,
            messages=messages,
            turns=turns,
            conversation_id=conversation_id,
        )

    def _build_messages(
        self,
        task: str,
        final_output: str,
        tool_calls: list[LCSToolCall],
        raw_tool_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build OpenAI-format messages from LCS response data.

        Reconstructs a plausible conversation history for DeepEval.
        """
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": task},
        ]

        if tool_calls:
            # Group tool calls by round
            results_by_round: dict[int, list[tuple[LCSToolCall, str]]] = {}
            for tc in tool_calls:
                # Find the round for this tool call
                tc_round = 0
                for tr in raw_tool_results:
                    if tr.get("id") == tc.call_id:
                        tc_round = tr.get("round", 0)
                        break
                results_by_round.setdefault(tc_round, []).append((tc, tc.result))

            for round_num in sorted(results_by_round.keys()):
                round_calls = results_by_round[round_num]

                # Assistant message with tool calls
                assistant_tool_calls = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": str(tc.arguments),
                        },
                    }
                    for tc, _ in round_calls
                ]
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": assistant_tool_calls,
                })

                # Tool result messages
                for tc, result in round_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.call_id,
                        "content": result,
                    })

        # Final assistant message
        if final_output:
            messages.append({
                "role": "assistant",
                "content": final_output,
            })

        return messages
