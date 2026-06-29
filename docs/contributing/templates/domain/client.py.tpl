"""Client for {{DOMAIN_DESCRIPTION}} operations."""

from typing import TYPE_CHECKING, Any

from rhoai_mcp.domains.{{DOMAIN_NAME}}.models import {{RESOURCE_CLASS}}

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient


class {{DOMAIN_CLASS}}Client:
    """Client for {{DOMAIN_NAME}} domain operations."""

    def __init__(self, k8s: "K8sClient") -> None:
        self._k8s = k8s

    def list_{{RESOURCE_NAME}}s(self, namespace: str) -> list[dict[str, Any]]:
        """List all {{RESOURCE_NAME}} resources in a namespace.

        Args:
            namespace: Kubernetes namespace.

        Returns:
            List of formatted resource dicts.
        """
        # --- Option A: CRD-based resources ---
        # from rhoai_mcp.domains.{{DOMAIN_NAME}}.crds import {{DOMAIN_CLASS}}CRDs
        # items = self._k8s.list_resources(
        #     {{DOMAIN_CLASS}}CRDs.{{CRD_KIND_UPPER}},
        #     namespace=namespace,
        # )

        # --- Option B: Core API resources ---
        # items = self._k8s.core_v1.list_namespaced_config_map(
        #     namespace=namespace, label_selector="your-label=value"
        # ).items

        # TODO: Replace with actual K8s API call.
        items: list[Any] = []

        results = []
        for obj in items:
            resource = {{RESOURCE_CLASS}}.from_k8s(obj)
            results.append(
                {
                    "name": resource.metadata.name,
                    "display_name": resource.display_name,
                    "status": resource.status.value,
                    "_source": resource.metadata.to_source_dict(),
                }
            )
        return results

    def get_{{RESOURCE_NAME}}(self, name: str, namespace: str) -> {{RESOURCE_CLASS}}:
        """Get a single {{RESOURCE_NAME}} by name.

        Args:
            name: Resource name.
            namespace: Kubernetes namespace.

        Returns:
            Parsed {{RESOURCE_CLASS}} instance.
        """
        # TODO: Replace with actual K8s API call.
        # CRD: obj = self._k8s.get_resource(CRDs.X, name, namespace)
        # Core: obj = self._k8s.core_v1.read_namespaced_config_map(name=name, namespace=namespace)
        raise NotImplementedError("Replace with K8s API call")
