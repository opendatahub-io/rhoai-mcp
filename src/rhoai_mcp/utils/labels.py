"""RHOAI label constants and helpers."""

from typing import Any


class RHOAILabels:
    """RHOAI-specific Kubernetes labels."""

    # Dashboard labels
    DASHBOARD = "opendatahub.io/dashboard"

    # Model serving labels
    MODELMESH_ENABLED = "modelmesh-enabled"

    # Notebook labels
    NOTEBOOK_NAME = "notebook-name"

    # Data connection labels
    DATA_CONNECTION_AWS = "opendatahub.io/managed"

    # Component labels
    APP_KUBERNETES_COMPONENT = "app.kubernetes.io/component"
    APP_KUBERNETES_PART_OF = "app.kubernetes.io/part-of"
    APP_KUBERNETES_CREATED_BY = "app.kubernetes.io/created-by"

    # Managed-by labels
    APP_KUBERNETES_MANAGED_BY = "app.kubernetes.io/managed-by"
    MANAGED_BY_VALUE = "rhoai-mcp"

    # KServe labels
    KSERVE_INFERENCE_SERVICE = "serving.kserve.io/inferenceservice"

    @classmethod
    def dashboard_project_labels(cls) -> dict[str, str]:
        """Create labels for a Data Science Project namespace."""
        return {cls.DASHBOARD: "true"}

    @classmethod
    def is_dashboard_project(cls, labels: dict[str, Any] | None) -> bool:
        """Check if namespace is a Data Science Project."""
        if not labels:
            return False
        return labels.get(cls.DASHBOARD) == "true"

    @classmethod
    def model_serving_labels(cls, single_model: bool = True) -> dict[str, str]:
        """Create labels for model serving mode."""
        return {cls.MODELMESH_ENABLED: "false" if single_model else "true"}

    @classmethod
    def is_modelmesh_enabled(cls, labels: dict[str, Any] | None) -> bool:
        """Check if namespace has ModelMesh enabled (multi-model serving)."""
        if not labels:
            return False
        return labels.get(cls.MODELMESH_ENABLED) == "true"

    @classmethod
    def notebook_labels(cls, notebook_name: str) -> dict[str, str]:
        """Create labels for notebook resources."""
        return {
            cls.NOTEBOOK_NAME: notebook_name,
            cls.APP_KUBERNETES_CREATED_BY: "odh-notebook-controller",
        }

    @classmethod
    def data_connection_labels(cls) -> dict[str, str]:
        """Create labels for data connection secrets."""
        return {cls.DASHBOARD: "true"}

    @classmethod
    def managed_by_mcp_labels(cls) -> dict[str, str]:
        """Return the managed-by label pair for resources created by this MCP server."""
        return {cls.APP_KUBERNETES_MANAGED_BY: cls.MANAGED_BY_VALUE}

    @classmethod
    def is_managed_by_mcp(cls, labels: dict[str, Any] | None) -> bool:
        """Check if a resource was created by this MCP server."""
        if not labels:
            return False
        return labels.get(cls.APP_KUBERNETES_MANAGED_BY) == cls.MANAGED_BY_VALUE

    @classmethod
    def filter_selector(cls, **labels: str) -> str:
        """Create a label selector string from key-value pairs."""
        return ",".join(f"{k}={v}" for k, v in labels.items())
