"""Models and validation for explicitly exposed AutoML deployments."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

AUTO_ML_EXPOSED_LABEL = "automl.opendatahub.io/mcp-exposed"
AUTO_ML_SCHEMA_ANNOTATION = "automl.opendatahub.io/input-schema"


class AutoMLTaskType(StrEnum):
    """AutoML tasks supported by the initial MCP surface."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class AutoMLFeature(BaseModel):
    """One JSON-record feature accepted by an AutoML deployment."""

    name: str = Field(min_length=1)
    json_type: str = Field(pattern="^(boolean|integer|number|string)$")
    nullable: bool = False


class AutoMLSchema(BaseModel):
    """Versioned, deployment-side input contract for an AutoML predictor."""

    schema_version: str = "v1"
    task_type: AutoMLTaskType
    features: list[AutoMLFeature] = Field(min_length=1)
    example_record: dict[str, Any]
    class_labels: list[str] | None = None

    @model_validator(mode="after")
    def classification_has_labels(self) -> AutoMLSchema:
        if self.task_type == AutoMLTaskType.CLASSIFICATION and not self.class_labels:
            raise ValueError("classification schemas must include class_labels")
        return self

    def records_to_v2_request(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate JSON records and encode them as a KServe v2 request."""
        if not records:
            raise ValueError("records must contain at least one row")

        inputs = []
        for feature in self.features:
            values = []
            for row_index, record in enumerate(records):
                if feature.name not in record:
                    raise ValueError(
                        f"records[{row_index}] is missing required feature '{feature.name}'"
                    )
                value = record[feature.name]
                if value is None:
                    if not feature.nullable:
                        raise ValueError(f"records[{row_index}].{feature.name} cannot be null")
                elif not _matches_json_type(value, feature.json_type):
                    raise ValueError(
                        f"records[{row_index}].{feature.name} must be {feature.json_type}"
                    )
                values.append(value)
            inputs.append(
                {
                    "name": feature.name,
                    "shape": [len(records)],
                    "datatype": _KSERVE_DATATYPES[feature.json_type],
                    "data": values,
                }
            )
        return {"inputs": inputs}


class AutoMLDeployment(BaseModel):
    """A discovered AutoGluon InferenceService that opted into MCP exposure."""

    name: str
    namespace: str
    status: str
    url: str | None = None
    input_schema: AutoMLSchema | None = Field(default=None, serialization_alias="schema")

    @property
    def is_ready(self) -> bool:
        return self.status == "Ready"

    @classmethod
    def from_inference_service(cls, resource: Any) -> AutoMLDeployment | None:
        """Parse an eligible KServe InferenceService, or return ``None``."""
        metadata = _field(resource, "metadata", {})
        labels = _field(metadata, "labels", {}) or {}
        annotations = _field(metadata, "annotations", {}) or {}
        model = _field(_field(_field(resource, "spec", {}), "predictor", {}), "model", {})
        model_format = _field(_field(model, "modelFormat", {}), "name")
        if labels.get(AUTO_ML_EXPOSED_LABEL) != "true" or model_format != "autogluon":
            return None

        schema_value = annotations.get(AUTO_ML_SCHEMA_ANNOTATION)
        schema = AutoMLSchema.model_validate_json(schema_value) if schema_value else None
        status = _field(resource, "status", {})
        conditions = _field(status, "conditions", []) or []
        ready = next((c for c in conditions if _field(c, "type") == "Ready"), None)
        return cls(
            name=_field(metadata, "name"),
            namespace=_field(metadata, "namespace"),
            status="Ready" if _field(ready, "status") == "True" else "NotReady",
            url=_field(status, "url"),
            input_schema=schema,
        )


_KSERVE_DATATYPES = {"boolean": "BOOL", "integer": "INT64", "number": "FP64", "string": "BYTES"}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _matches_json_type(value: Any, json_type: str) -> bool:
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, str)
