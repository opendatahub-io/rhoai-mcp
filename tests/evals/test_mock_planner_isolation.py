"""Exercise the mock server's embedded planner without cluster or LLM access."""

import asyncio
import json
import sys
from pathlib import Path
from textwrap import dedent

import pytest

pytest.importorskip("pydantic_ai", reason="requires the skill-eval dependency group")

from evals.agent.models import AgentModelSpec, get_spec  # noqa: E402
from evals.agent.runner import build_agent  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402


async def test_mock_subprocess_recommends_without_cluster_or_network_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PLANNER_DETECT_CLUSTER_GPUS", "true")
    monkeypatch.setattr(AgentModelSpec, "build", lambda _self: TestModel())
    _agent, toolset = build_agent(get_spec("enmaas-glm"))
    transport = toolset.client.transport
    audit_path = tmp_path / "probe-attempts.json"
    wrapper = tmp_path / "guarded_server.py"
    wrapper.write_text(
        dedent(
            f"""\
            import json
            from pathlib import Path
            import socket
            import sys
            from planner.cluster import gpu_detector

            audit_path = Path({str(audit_path)!r})
            attempts = dict(config_loads=0, node_lists=0, network=0)
            audit_path.write_text(json.dumps(attempts))

            def forbidden(name):
                attempts[name] += 1
                audit_path.write_text(json.dumps(attempts))
                raise AssertionError(name + ' is forbidden in this offline test')

            gpu_detector.reset_gpu_cache()
            gpu_detector._load_k8s_config = lambda: forbidden('config_loads')
            gpu_detector._list_nodes = lambda: forbidden('node_lists')
            socket.socket.connect = lambda *a, **kw: forbidden('network')
            socket.getaddrinfo = lambda *a, **kw: forbidden('network')

            from rhoai_mcp.__main__ import main
            sys.argv = ['rhoai-mcp', *sys.argv[3:]]
            raise SystemExit(main())
            """
        )
    )
    # Keep the runner's arguments and environment; launch its CLI through guards
    # using the installed interpreter, without asking uv to resolve dependencies.
    transport.command = sys.executable
    transport.args = [str(wrapper), *transport.args]
    transport.env = {
        **(transport.env or {}),
        "RHOAI_MCP_PLANNER_MODE": "local",
        "RHOAI_MCP_PLANNER_MODEL_CATALOG_URL": "",
    }

    async with asyncio.timeout(30), toolset.client:
        response = await toolset.client.call_tool(
            "recommend_model",
            {
                "text": "Support chatbot",
                "use_case": "chatbot_conversational",
                "user_count": 50,
                "preferred_gpu_types": ["A100-80"],
            },
        )

    assert not response.is_error
    payload = response.structured_content
    assert payload is not None and "error" not in payload
    recommendations = payload["recommendations"]
    assert len(recommendations) == 4
    assert all(rec["gpu"].endswith(" A100-80") for rec in recommendations.values())
    assert json.loads(audit_path.read_text()) == {"config_loads": 0, "node_lists": 0, "network": 0}
