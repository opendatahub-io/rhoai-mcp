"""Client for the example domain (ConfigMap-backed)."""

from typing import TYPE_CHECKING, Any

from rhoai_mcp.domains._example.models import ExampleItem

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient

EXAMPLE_LABEL_SELECTOR = "rhoai.io/example=true"


class ExampleClient:
    """Client for example domain operations.

    Reads ConfigMaps labelled with rhoai.io/example=true as a
    demonstration of the domain client pattern. Uses the core
    Kubernetes API — no custom CRDs required.
    """

    def __init__(self, k8s: "K8sClient") -> None:
        self._k8s = k8s

    def list_items(self, namespace: str) -> list[dict[str, Any]]:
        """List example items in a namespace."""
        cm_list = self._k8s.core_v1.list_namespaced_config_map(
            namespace=namespace, label_selector=EXAMPLE_LABEL_SELECTOR
        )

        results = []
        for cm in cm_list.items:
            item = ExampleItem.from_configmap(cm)
            results.append(
                {
                    "name": item.metadata.name,
                    "display_name": item.display_name,
                    "data": item.data,
                    "status": item.status.value,
                    "_source": item.metadata.to_source_dict(),
                }
            )
        return results

    def get_item(self, name: str, namespace: str) -> ExampleItem:
        """Get a single example item by name."""
        cm = self._k8s.core_v1.read_namespaced_config_map(name=name, namespace=namespace)
        return ExampleItem.from_configmap(cm)
