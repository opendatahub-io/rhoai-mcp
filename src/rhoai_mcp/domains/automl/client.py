"""Kubernetes discovery for exposed AutoGluon InferenceServices."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, cast

import httpx

from rhoai_mcp.domains.automl.models import AUTO_ML_EXPOSED_LABEL, AutoMLDeployment
from rhoai_mcp.domains.inference.crds import InferenceCRDs

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient

logger = logging.getLogger(__name__)


class AutoMLClient:
    """Read-only client for AutoML deployments opted into MCP exposure."""

    def __init__(self, k8s: K8sClient) -> None:
        self._k8s = k8s

    def list_deployments(self, namespace: str) -> list[AutoMLDeployment]:
        resources = self._k8s.list_resources(
            InferenceCRDs.INFERENCE_SERVICE,
            namespace=namespace,
            label_selector=f"{AUTO_ML_EXPOSED_LABEL}=true",
        )
        deployments = []
        for item in resources:
            try:
                deployment = AutoMLDeployment.from_inference_service(item)
            except ValueError as error:
                logger.warning("Skipping AutoML deployment with an invalid input schema: %s", error)
                continue
            if deployment is not None:
                deployments.append(deployment)
        return deployments

    def get_deployment(self, name: str, namespace: str) -> AutoMLDeployment:
        resource = self._k8s.get(InferenceCRDs.INFERENCE_SERVICE, name=name, namespace=namespace)
        deployment = AutoMLDeployment.from_inference_service(resource)
        if deployment is None:
            raise ValueError(f"InferenceService '{name}' is not an exposed AutoML deployment")
        return deployment

    async def predict_deployment(
        self, name: str, namespace: str, records: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Validate records, invoke KServe v2, and normalize the prediction response."""
        deployment = self.get_deployment(name, namespace)
        if deployment.input_schema is None:
            raise ValueError("Deployment has no AutoML MCP input schema annotation")
        if not deployment.is_ready:
            raise ValueError(f"Deployment '{name}' is not ready for predictions")
        if not deployment.url:
            raise ValueError(f"Deployment '{name}' has no prediction endpoint URL")

        request_body = deployment.input_schema.records_to_v2_request(records)
        endpoint = f"{deployment.url.rstrip('/')}/v2/models/{name}/infer"

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                response = await http_client.post(endpoint, json=request_body)
        except httpx.TimeoutException as error:
            raise ValueError(f"Prediction request to '{name}' timed out") from error
        except httpx.RequestError as error:
            raise ValueError(
                f"Could not reach prediction endpoint for '{name}': {error}"
            ) from error

        if response.is_error:
            raise ValueError(_prediction_error_message(response, name))

        try:
            response_body = response.json()
        except ValueError as error:
            raise ValueError(f"Prediction endpoint for '{name}' returned invalid JSON") from error

        return _normalize_prediction_response(deployment, response_body)


def _prediction_error_message(response: httpx.Response, name: str) -> str:
    """Return a concise, agent-usable KServe endpoint error."""
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = body.get("error") or body.get("message") or body
        else:
            detail = body
    except ValueError:
        detail = response.text
    detail_text = str(detail).strip().replace("\n", " ")[:500]
    return f"Prediction endpoint for '{name}' returned HTTP {response.status_code}: {detail_text}"


def _normalize_prediction_response(deployment: AutoMLDeployment, response: Any) -> dict[str, Any]:
    """Convert a KServe v2 response into task-aware, agent-friendly predictions."""
    if not isinstance(response, dict) or not isinstance(response.get("outputs"), list):
        raise ValueError("Prediction endpoint returned no KServe v2 outputs")

    outputs = response["outputs"]
    if deployment.input_schema is None:  # Defensive: checked before the HTTP request.
        raise ValueError("Deployment has no AutoML MCP input schema annotation")

    if deployment.input_schema.task_type == "regression":
        prediction_output = _output_named(outputs, "predictions")
        values = _output_values(prediction_output)
        return {
            "task_type": "regression",
            "predictions": [{"value": value} for value in values],
        }

    probability_outputs = [
        output for output in outputs if _output_name(output).startswith("proba_")
    ]
    if probability_outputs:
        return _classification_probabilities(deployment, probability_outputs)

    prediction_output = _output_named(outputs, "predictions")
    values = _output_values(prediction_output)
    return {
        "task_type": "classification",
        "predictions": [{"class": str(value), "probabilities": None} for value in values],
        "probabilities_available": False,
        "message": "The serving runtime returned class labels only; enable PREDICT_PROBA=true "
        "in the AutoGluon ServingRuntime to return class probabilities.",
    }


def _classification_probabilities(
    deployment: AutoMLDeployment, outputs: list[Any]
) -> dict[str, Any]:
    """Convert AutoGluon ``proba_<class>`` outputs to per-record probabilities."""
    labels = deployment.input_schema.class_labels if deployment.input_schema else None
    if not labels:
        raise ValueError(
            "Prediction endpoint probability outputs do not match the schema class labels"
        )

    outputs_by_name: dict[str, Any] = {}
    for output in outputs:
        name = _output_name(output)
        if name in outputs_by_name:
            raise ValueError(f"Prediction endpoint returned duplicate probability output '{name}'")
        outputs_by_name[name] = output

    expected_names_by_label = {label: _probability_output_name(label) for label in labels}
    expected_names = set(expected_names_by_label.values())
    if len(expected_names) != len(labels):
        raise ValueError("Schema class labels map to duplicate probability output names")
    if set(outputs_by_name) != expected_names:
        raise ValueError(
            "Prediction endpoint probability outputs do not match the schema class labels"
        )

    probabilities_by_class = {
        label: _probability_values(outputs_by_name[expected_names_by_label[label]])
        for label in labels
    }
    row_count = len(next(iter(probabilities_by_class.values())))
    if any(len(values) != row_count for values in probabilities_by_class.values()):
        raise ValueError(
            "Prediction endpoint returned probability outputs with inconsistent lengths"
        )

    predictions = []
    for row_index in range(row_count):
        probabilities = {
            label: values[row_index] for label, values in probabilities_by_class.items()
        }
        predictions.append(
            {
                "class": max(probabilities.items(), key=lambda item: item[1])[0],
                "probabilities": probabilities,
            }
        )

    return {
        "task_type": "classification",
        "predictions": predictions,
        "probabilities_available": True,
    }


def _output_named(outputs: list[Any], name: str) -> Any:
    output = next((item for item in outputs if _output_name(item) == name), None)
    if output is None:
        raise ValueError(f"Prediction endpoint returned no '{name}' output")
    return output


def _output_name(output: Any) -> str:
    if not isinstance(output, dict) or not isinstance(output.get("name"), str):
        raise ValueError("Prediction endpoint returned an invalid KServe output")
    return cast(str, output["name"])


def _output_values(output: Any) -> list[Any]:
    if not isinstance(output, dict) or not isinstance(output.get("data"), list):
        raise ValueError(f"Prediction endpoint output '{_output_name(output)}' has no data array")
    return cast(list[Any], output["data"])


def _probability_values(output: Any) -> list[float]:
    """Return numeric probability data and reject malformed runtime responses."""
    values = _output_values(output)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
        raise ValueError(
            f"Prediction endpoint output '{_output_name(output)}' has non-numeric probabilities"
        )
    return [float(value) for value in values]


def _probability_output_name(label: str) -> str:
    """Return AutoGluon's KServe v2 tensor name for a class label."""
    sanitized_label = re.sub(r"[^0-9A-Za-z_]", "_", label)
    if sanitized_label and sanitized_label[0].isdigit():
        sanitized_label = f"_{sanitized_label}"
    return f"proba_{sanitized_label}"
