"""Tests for exposed AutoML deployment discovery."""

from unittest.mock import MagicMock

from rhoai_mcp.domains.automl.client import AutoMLClient
from tests.domains.automl.test_models import _resource


def test_list_uses_opt_in_label_and_rechecks_model_format() -> None:
    k8s = MagicMock()
    k8s.list_resources.return_value = [_resource(), _resource(model_format="sklearn")]

    deployments = AutoMLClient(k8s).list_deployments("demo")

    assert [item.name for item in deployments] == ["churn"]
    assert (
        k8s.list_resources.call_args.kwargs["label_selector"]
        == "automl.opendatahub.io/mcp-exposed=true"
    )
