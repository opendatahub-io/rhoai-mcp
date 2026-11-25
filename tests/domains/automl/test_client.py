"""Tests for exposed AutoML deployment discovery and prediction."""

import json
import logging
from unittest.mock import MagicMock

import httpx
import pytest

from rhoai_mcp.domains.automl.client import (
    AutoMLClient,
    _normalize_prediction_response,
    _prediction_error_message,
)
from rhoai_mcp.domains.automl.models import AutoMLDeployment
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


def test_list_skips_deployment_with_invalid_schema(caplog: pytest.LogCaptureFixture) -> None:
    k8s = MagicMock()
    invalid = _resource()
    invalid_schema = json.loads(invalid.metadata.annotations["automl.opendatahub.io/input-schema"])
    invalid_schema.pop("class_labels")
    invalid.metadata.annotations["automl.opendatahub.io/input-schema"] = json.dumps(invalid_schema)
    k8s.list_resources.return_value = [invalid, _resource()]

    with caplog.at_level(logging.WARNING):
        deployments = AutoMLClient(k8s).list_deployments("demo")

    assert [item.name for item in deployments] == ["churn"]
    assert "Skipping AutoML deployment with an invalid input schema" in caplog.text


def test_prediction_error_handles_structured_json_body() -> None:
    response = httpx.Response(
        503,
        json={"error": {"message": "unavailable"}},
        request=httpx.Request("POST", "http://churn"),
    )

    assert _prediction_error_message(response, "churn") == (
        "Prediction endpoint for 'churn' returned HTTP 503: {'message': 'unavailable'}"
    )


def test_prediction_error_handles_json_array_body() -> None:
    response = httpx.Response(
        503,
        json=["unavailable"],
        request=httpx.Request("POST", "http://churn"),
    )

    assert _prediction_error_message(response, "churn") == (
        "Prediction endpoint for 'churn' returned HTTP 503: ['unavailable']"
    )


def test_normalizes_classification_probabilities() -> None:
    deployment = AutoMLDeployment.from_inference_service(_resource())
    assert deployment is not None

    result = _normalize_prediction_response(
        deployment,
        {
            "outputs": [
                {"name": "proba_no", "data": [0.8, 0.2]},
                {"name": "proba_yes", "data": [0.2, 0.8]},
            ]
        },
    )

    assert result == {
        "task_type": "classification",
        "predictions": [
            {"class": "no", "probabilities": {"no": 0.8, "yes": 0.2}},
            {"class": "yes", "probabilities": {"no": 0.2, "yes": 0.8}},
        ],
        "probabilities_available": True,
    }


def test_normalizes_probability_outputs_by_name_not_order() -> None:
    deployment = AutoMLDeployment.from_inference_service(_resource())
    assert deployment is not None

    result = _normalize_prediction_response(
        deployment,
        {
            "outputs": [
                {"name": "proba_yes", "data": [0.9]},
                {"name": "proba_no", "data": [0.1]},
            ]
        },
    )

    assert result["predictions"] == [{"class": "yes", "probabilities": {"no": 0.1, "yes": 0.9}}]


def test_normalizes_numeric_class_label_probability_outputs() -> None:
    deployment = AutoMLDeployment.from_inference_service(_resource())
    assert deployment is not None
    assert deployment.input_schema is not None
    deployment.input_schema.class_labels = ["0", "1"]

    result = _normalize_prediction_response(
        deployment,
        {
            "outputs": [
                {"name": "proba__1", "data": [0.75]},
                {"name": "proba__0", "data": [0.25]},
            ]
        },
    )

    assert result["predictions"] == [{"class": "1", "probabilities": {"0": 0.25, "1": 0.75}}]


def test_rejects_duplicate_probability_output_names() -> None:
    deployment = AutoMLDeployment.from_inference_service(_resource())
    assert deployment is not None

    with pytest.raises(ValueError, match="duplicate probability output 'proba_no'"):
        _normalize_prediction_response(
            deployment,
            {
                "outputs": [
                    {"name": "proba_no", "data": [0.1]},
                    {"name": "proba_no", "data": [0.2]},
                ]
            },
        )


def test_rejects_schema_labels_that_map_to_the_same_probability_output() -> None:
    deployment = AutoMLDeployment.from_inference_service(_resource())
    assert deployment is not None
    assert deployment.input_schema is not None
    deployment.input_schema.class_labels = ["a-b", "a_b"]

    with pytest.raises(
        ValueError, match="Schema class labels map to duplicate probability output names"
    ):
        _normalize_prediction_response(
            deployment,
            {
                "outputs": [
                    {"name": "proba_a_b", "data": [0.1]},
                ]
            },
        )


def test_normalizes_label_only_classification_response() -> None:
    deployment = AutoMLDeployment.from_inference_service(_resource())
    assert deployment is not None

    result = _normalize_prediction_response(
        deployment, {"outputs": [{"name": "predictions", "data": ["yes"]}]}
    )

    assert result["predictions"] == [{"class": "yes", "probabilities": None}]
    assert result["probabilities_available"] is False


@pytest.mark.asyncio
async def test_prediction_posts_kserve_v2_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    k8s = MagicMock()
    k8s.get.return_value = _resource()
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> httpx.Response:
            calls.append((url, json))
            return httpx.Response(
                200,
                json={"outputs": [{"name": "predictions", "data": ["yes"]}]},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("rhoai_mcp.domains.automl.client.httpx.AsyncClient", FakeAsyncClient)

    result = await AutoMLClient(k8s).predict_deployment(
        "churn", "demo", [{"age": 42, "country": "PL"}]
    )

    assert calls == [
        (
            "http://churn/v2/models/churn/infer",
            {
                "inputs": [
                    {"name": "age", "shape": [1], "datatype": "INT64", "data": [42]},
                    {
                        "name": "country",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": ["PL"],
                    },
                ]
            },
        )
    ]
    assert result["predictions"] == [{"class": "yes", "probabilities": None}]
