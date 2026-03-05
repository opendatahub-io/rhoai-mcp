"""Mock Kubernetes client for running rhoai-mcp without a real cluster.

Provides MockK8sClient and cluster state utilities used by the eval
framework and the --mock-cluster server mode.
"""

from rhoai_mcp.mock_k8s.cluster_state import ClusterState, create_default_cluster_state
from rhoai_mcp.mock_k8s.mock_client import MockK8sClient

__all__ = ["ClusterState", "MockK8sClient", "create_default_cluster_state"]
