"""K8s RBAC permission mappings for composite plugin tools.

Each mapping declares which Kubernetes API permissions a tool requires.
Keys are tool function names (matching @mcp.tool() decorated functions),
values are lists of permission dicts with apiGroup, resource, and verb.
"""

CLUSTER_PERMISSIONS: dict[str, list[dict[str, str]]] = {
    "cluster_summary": [
        {"apiGroup": "project.openshift.io", "resource": "projects", "verb": "list"},
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "list"},
        {"apiGroup": "serving.kserve.io", "resource": "inferenceservices", "verb": "list"},
    ],
    "project_summary": [
        {"apiGroup": "project.openshift.io", "resource": "projects", "verb": "get"},
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "list"},
        {"apiGroup": "serving.kserve.io", "resource": "inferenceservices", "verb": "list"},
    ],
    "resource_status": [
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "get"},
        {"apiGroup": "serving.kserve.io", "resource": "inferenceservices", "verb": "get"},
        {"apiGroup": "", "resource": "persistentvolumeclaims", "verb": "get"},
    ],
    "list_resource_names": [
        {"apiGroup": "project.openshift.io", "resource": "projects", "verb": "list"},
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "list"},
        {"apiGroup": "serving.kserve.io", "resource": "inferenceservices", "verb": "list"},
        {"apiGroup": "", "resource": "persistentvolumeclaims", "verb": "list"},
    ],
    "multi_resource_status": [
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "get"},
        {"apiGroup": "serving.kserve.io", "resource": "inferenceservices", "verb": "get"},
        {"apiGroup": "", "resource": "persistentvolumeclaims", "verb": "get"},
    ],
    "explore_cluster": [
        {"apiGroup": "project.openshift.io", "resource": "projects", "verb": "list"},
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "list"},
        {"apiGroup": "serving.kserve.io", "resource": "inferenceservices", "verb": "list"},
        {"apiGroup": "", "resource": "nodes", "verb": "list"},
    ],
    "diagnose_resource": [
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "get"},
        {"apiGroup": "serving.kserve.io", "resource": "inferenceservices", "verb": "get"},
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "get"},
        {"apiGroup": "", "resource": "events", "verb": "list"},
        {"apiGroup": "", "resource": "pods", "verb": "list"},
        {"apiGroup": "", "resource": "pods/log", "verb": "get"},
    ],
    "get_resource": [
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "get"},
        {"apiGroup": "serving.kserve.io", "resource": "inferenceservices", "verb": "get"},
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "get"},
        {"apiGroup": "", "resource": "persistentvolumeclaims", "verb": "get"},
        {"apiGroup": "", "resource": "secrets", "verb": "get"},
    ],
    "list_resources": [
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "list"},
        {"apiGroup": "serving.kserve.io", "resource": "inferenceservices", "verb": "list"},
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "list"},
        {"apiGroup": "", "resource": "persistentvolumeclaims", "verb": "list"},
        {"apiGroup": "", "resource": "secrets", "verb": "list"},
    ],
    "manage_resource": [
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "patch"},
        {"apiGroup": "serving.kserve.io", "resource": "inferenceservices", "verb": "delete"},
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "patch"},
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "delete"},
    ],
}

TRAINING_COMPOSITE_PERMISSIONS: dict[str, list[dict[str, str]]] = {
    # planning tools
    "estimate_resources": [
        {"apiGroup": "", "resource": "nodes", "verb": "list"},
    ],
    "check_training_prerequisites": [
        {"apiGroup": "", "resource": "nodes", "verb": "list"},
        {"apiGroup": "kubeflow.org", "resource": "clustertrainingruntimes", "verb": "list"},
        {"apiGroup": "", "resource": "persistentvolumeclaims", "verb": "get"},
    ],
    "validate_training_config": [
        {"apiGroup": "kubeflow.org", "resource": "clustertrainingruntimes", "verb": "get"},
        {"apiGroup": "", "resource": "persistentvolumeclaims", "verb": "get"},
    ],
    "setup_hf_credentials": [
        {"apiGroup": "", "resource": "secrets", "verb": "create"},
        {"apiGroup": "", "resource": "secrets", "verb": "delete"},
    ],
    "prepare_training": [
        {"apiGroup": "", "resource": "nodes", "verb": "list"},
        {"apiGroup": "kubeflow.org", "resource": "clustertrainingruntimes", "verb": "list"},
        {"apiGroup": "", "resource": "persistentvolumeclaims", "verb": "get"},
        {"apiGroup": "", "resource": "persistentvolumeclaims", "verb": "create"},
    ],
    # storage tools
    "setup_training_storage": [
        {"apiGroup": "", "resource": "persistentvolumeclaims", "verb": "create"},
        {"apiGroup": "", "resource": "persistentvolumeclaims", "verb": "get"},
    ],
    "setup_nfs_storage": [
        {"apiGroup": "storage.k8s.io", "resource": "storageclasses", "verb": "list"},
        {"apiGroup": "", "resource": "persistentvolumes", "verb": "create"},
    ],
    "fix_pvc_permissions": [
        {"apiGroup": "", "resource": "persistentvolumeclaims", "verb": "get"},
        {"apiGroup": "", "resource": "pods", "verb": "create"},
    ],
    # unified training tool
    "training": [
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "list"},
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "get"},
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "create"},
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "patch"},
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "delete"},
        {"apiGroup": "kubeflow.org", "resource": "clustertrainingruntimes", "verb": "list"},
        {"apiGroup": "", "resource": "pods/log", "verb": "get"},
        {"apiGroup": "", "resource": "events", "verb": "list"},
    ],
}
