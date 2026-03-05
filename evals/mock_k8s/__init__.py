"""Shim re-exporting mock_k8s from its canonical location in rhoai_mcp."""

from rhoai_mcp.mock_k8s import ClusterState, MockK8sClient, create_default_cluster_state

__all__ = ["ClusterState", "MockK8sClient", "create_default_cluster_state"]
