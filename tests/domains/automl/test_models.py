"""Tests for AutoML schema parsing and KServe payload conversion."""

from types import SimpleNamespace

import pytest

from rhoai_mcp.domains.automl.models import (
    AUTO_ML_EXPOSED_LABEL,
    AUTO_ML_SCHEMA_ANNOTATION,
    AutoMLDeployment,
    AutoMLSchema,
)


def _resource(*, exposed: bool = True, model_format: str = "autogluon") -> SimpleNamespace:
    schema = {
        "task_type": "classification",
        "features": [
            {"name": "age", "json_type": "integer"},
            {"name": "country", "json_type": "string"},
        ],
        "example_record": {"age": 42, "country": "PL"},
        "class_labels": ["no", "yes"],
    }
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name="churn",
            namespace="demo",
            labels={AUTO_ML_EXPOSED_LABEL: "true"} if exposed else {},
            annotations={AUTO_ML_SCHEMA_ANNOTATION: __import__("json").dumps(schema)},
        ),
        spec={"predictor": {"model": {"modelFormat": {"name": model_format}}}},
        status={"url": "http://churn", "conditions": [{"type": "Ready", "status": "True"}]},
    )


def test_only_explicitly_exposed_autogluon_service_is_a_deployment() -> None:
    deployment = AutoMLDeployment.from_inference_service(_resource())

    assert deployment is not None
    assert deployment.name == "churn"
    assert deployment.input_schema is not None
    assert deployment.input_schema.task_type == "classification"
    assert AutoMLDeployment.from_inference_service(_resource(exposed=False)) is None
    assert AutoMLDeployment.from_inference_service(_resource(model_format="sklearn")) is None


def test_records_become_named_kserve_v2_tensors() -> None:
    schema = AutoMLDeployment.from_inference_service(_resource()).input_schema
    assert schema is not None

    assert schema.records_to_v2_request([{"age": 42, "country": "PL"}]) == {
        "inputs": [
            {"name": "age", "shape": [1], "datatype": "INT64", "data": [42]},
            {"name": "country", "shape": [1], "datatype": "BYTES", "data": ["PL"]},
        ]
    }


def test_records_return_agent_usable_validation_errors() -> None:
    schema = AutoMLDeployment.from_inference_service(_resource()).input_schema
    assert schema is not None

    with pytest.raises(ValueError, match="missing required feature 'country'"):
        schema.records_to_v2_request([{"age": 42}])
    with pytest.raises(ValueError, match="age must be integer"):
        schema.records_to_v2_request([{"age": "forty-two", "country": "PL"}])


def test_classifier_schema_requires_class_labels() -> None:
    with pytest.raises(ValueError, match="class_labels"):
        AutoMLSchema.model_validate(
            {
                "task_type": "classification",
                "features": [{"name": "x", "json_type": "number"}],
                "example_record": {"x": 1},
            }
        )
