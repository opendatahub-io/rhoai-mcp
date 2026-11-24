"""Subprocess workload for the local planner concurrency integration test."""

import asyncio
from typing import Any


async def main() -> None:
    from rhoai_mcp.composites.planner.local_client import LocalPlannerClient

    client = LocalPlannerClient()
    common: dict[str, Any] = {
        "text": "customer support chatbot",
        "use_case_override": "chatbot_conversational",
        "user_count_override": 50,
        "gpu_types_override": ["A100-80"],
    }
    await client.recommend(**common)
    for _ in range(10):
        results = await asyncio.gather(
            client.recommend(**common, max_cost=1788.5),
            client.recommend(**common, ttft_override_ms=44),
            client.recommend(**common, min_quality=33),
        )
        assert all(result.specification["user_count"] == 50 for result in results)


if __name__ == "__main__":
    # A native crash should not leave a core dump in the checkout.
    try:
        import resource
    except ImportError:  # Windows
        pass
    else:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    asyncio.run(main())
