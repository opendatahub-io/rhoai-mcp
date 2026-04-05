"""K8s RBAC permission mappings for domain plugin tools.

Each mapping declares which Kubernetes API permissions a tool requires.
Keys are tool function names (matching @mcp.tool() decorated functions),
values are lists of permission dicts with apiGroup, resource, and verb.
"""

PROJECTS_PERMISSIONS: dict[str, list[dict[str, str]]] = {
    "list_data_science_projects": [
        {"apiGroup": "project.openshift.io", "resource": "projects", "verb": "list"},
    ],
    "get_project_details": [
        {"apiGroup": "project.openshift.io", "resource": "projects", "verb": "get"},
    ],
    "create_data_science_project": [
        {"apiGroup": "", "resource": "namespaces", "verb": "create"},
    ],
    "delete_data_science_project": [
        {"apiGroup": "project.openshift.io", "resource": "projects", "verb": "delete"},
    ],
    "set_model_serving_mode": [
        {"apiGroup": "", "resource": "namespaces", "verb": "patch"},
    ],
}

NOTEBOOKS_PERMISSIONS: dict[str, list[dict[str, str]]] = {
    "list_workbenches": [
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "list"},
    ],
    "get_workbench": [
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "get"},
    ],
    "create_workbench": [
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "create"},
    ],
    "start_workbench": [
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "patch"},
    ],
    "stop_workbench": [
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "patch"},
    ],
    "delete_workbench": [
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "delete"},
    ],
    "list_notebook_images": [
        {"apiGroup": "image.openshift.io", "resource": "imagestreams", "verb": "list"},
    ],
    "get_workbench_url": [
        {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "get"},
        {"apiGroup": "route.openshift.io", "resource": "routes", "verb": "get"},
    ],
}

INFERENCE_PERMISSIONS: dict[str, list[dict[str, str]]] = {
    "list_inference_services": [
        {"apiGroup": "serving.kserve.io", "resource": "inferenceservices", "verb": "list"},
    ],
    "get_inference_service": [
        {"apiGroup": "serving.kserve.io", "resource": "inferenceservices", "verb": "get"},
    ],
    "deploy_model": [
        {"apiGroup": "serving.kserve.io", "resource": "inferenceservices", "verb": "create"},
    ],
    "delete_inference_service": [
        {"apiGroup": "serving.kserve.io", "resource": "inferenceservices", "verb": "delete"},
    ],
    "list_serving_runtimes": [
        {"apiGroup": "serving.kserve.io", "resource": "servingruntimes", "verb": "list"},
    ],
    "create_serving_runtime": [
        {"apiGroup": "serving.kserve.io", "resource": "servingruntimes", "verb": "create"},
    ],
    "get_model_endpoint": [
        {"apiGroup": "serving.kserve.io", "resource": "inferenceservices", "verb": "get"},
    ],
    "prepare_model_deployment": [
        {"apiGroup": "serving.kserve.io", "resource": "servingruntimes", "verb": "list"},
        {"apiGroup": "", "resource": "persistentvolumeclaims", "verb": "list"},
        {"apiGroup": "", "resource": "secrets", "verb": "list"},
    ],
    "check_deployment_prerequisites": [
        {"apiGroup": "serving.kserve.io", "resource": "servingruntimes", "verb": "list"},
    ],
    "estimate_serving_resources": [
        {"apiGroup": "serving.kserve.io", "resource": "servingruntimes", "verb": "list"},
    ],
    "recommend_serving_runtime": [
        {"apiGroup": "serving.kserve.io", "resource": "servingruntimes", "verb": "list"},
    ],
    "test_model_endpoint": [
        {"apiGroup": "serving.kserve.io", "resource": "inferenceservices", "verb": "get"},
    ],
}

_DSPA = "datasciencepipelinesapplications.opendatahub.io"

PIPELINES_PERMISSIONS: dict[str, list[dict[str, str]]] = {
    "get_pipeline_server": [
        {"apiGroup": _DSPA, "resource": "datasciencepipelinesapplications", "verb": "get"},
    ],
    "create_pipeline_server": [
        {"apiGroup": _DSPA, "resource": "datasciencepipelinesapplications", "verb": "create"},
        {"apiGroup": "", "resource": "secrets", "verb": "create"},
    ],
    "delete_pipeline_server": [
        {"apiGroup": _DSPA, "resource": "datasciencepipelinesapplications", "verb": "delete"},
    ],
}

CONNECTIONS_PERMISSIONS: dict[str, list[dict[str, str]]] = {
    "list_data_connections": [
        {"apiGroup": "", "resource": "secrets", "verb": "list"},
    ],
    "get_data_connection": [
        {"apiGroup": "", "resource": "secrets", "verb": "get"},
    ],
    "create_s3_data_connection": [
        {"apiGroup": "", "resource": "secrets", "verb": "create"},
    ],
    "delete_data_connection": [
        {"apiGroup": "", "resource": "secrets", "verb": "delete"},
    ],
}

STORAGE_PERMISSIONS: dict[str, list[dict[str, str]]] = {
    "list_storage": [
        {"apiGroup": "", "resource": "persistentvolumeclaims", "verb": "list"},
    ],
    "create_storage": [
        {"apiGroup": "", "resource": "persistentvolumeclaims", "verb": "create"},
    ],
    "delete_storage": [
        {"apiGroup": "", "resource": "persistentvolumeclaims", "verb": "delete"},
    ],
}

QUICKSTARTS_PERMISSIONS: dict[str, list[dict[str, str]]] = {
    # list_available_quickstarts and get_quickstart_readme are ungoverned
    # (read from built-in registry / GitHub, no K8s API calls).
    "deploy_quickstart": [
        {"apiGroup": "apps", "resource": "deployments", "verb": "create"},
    ],
}

TRAINING_PERMISSIONS: dict[str, list[dict[str, str]]] = {
    # discovery tools
    "list_training_jobs": [
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "list"},
    ],
    "get_training_job": [
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "get"},
    ],
    "get_cluster_resources": [
        {"apiGroup": "", "resource": "nodes", "verb": "list"},
    ],
    "list_training_runtimes": [
        {"apiGroup": "kubeflow.org", "resource": "clustertrainingruntimes", "verb": "list"},
    ],
    # lifecycle tools
    "suspend_training_job": [
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "patch"},
    ],
    "resume_training_job": [
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "patch"},
    ],
    "delete_training_job": [
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "delete"},
    ],
    "wait_for_job_completion": [
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "get"},
    ],
    "get_job_spec": [
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "get"},
    ],
    # monitoring tools
    "get_training_progress": [
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "get"},
    ],
    "get_training_logs": [
        {"apiGroup": "", "resource": "pods", "verb": "list"},
        {"apiGroup": "", "resource": "pods/log", "verb": "get"},
    ],
    "get_job_events": [
        {"apiGroup": "", "resource": "events", "verb": "list"},
    ],
    "manage_checkpoints": [
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "get"},
    ],
    # runtime tools
    "get_runtime_details": [
        {"apiGroup": "kubeflow.org", "resource": "clustertrainingruntimes", "verb": "get"},
    ],
    "create_runtime": [
        {"apiGroup": "kubeflow.org", "resource": "clustertrainingruntimes", "verb": "create"},
    ],
    "setup_training_runtime": [
        {"apiGroup": "kubeflow.org", "resource": "clustertrainingruntimes", "verb": "create"},
    ],
    "delete_runtime": [
        {"apiGroup": "kubeflow.org", "resource": "clustertrainingruntimes", "verb": "delete"},
    ],
    # training tools
    "train": [
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "create"},
        {"apiGroup": "kubeflow.org", "resource": "clustertrainingruntimes", "verb": "list"},
    ],
    "run_container_training_job": [
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "create"},
    ],
    "analyze_training_failure": [
        {"apiGroup": "kubeflow.org", "resource": "trainjobs", "verb": "get"},
        {"apiGroup": "", "resource": "events", "verb": "list"},
        {"apiGroup": "", "resource": "pods/log", "verb": "get"},
    ],
}
