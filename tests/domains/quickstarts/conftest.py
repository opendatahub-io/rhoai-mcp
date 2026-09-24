"""Shared fixtures and sample artifacts for quickstarts domain tests."""

REGISTRY_YAML = b"""
apiVersion: quickstart.redhat.com/v1
kind: QuickstartRegistry
metadata:
  version: "1.0"
quickstarts:
  - name: peoplemesh
    displayName: "Peoplemesh"
    shortDescription: "Semantic talent search"
    latestManifestVersion: "1.0.0"
    availableManifestVersions:
      - version: "1.0.0"
        status: stable
    estimatedDeploymentTime: 15
    tags: ["llm", "rag"]
    industries: ["Human Resources"]
    manifestRepo: "quay.io/rh-ai-quickstart/peoplemesh-manifest"
"""

MANIFEST_YAML = b"""
apiVersion: quickstart.redhat.com/v1
kind: Quickstart
metadata:
  name: peoplemesh
  displayName: "Peoplemesh"
  version: "1.0.0"
  shortDescription: "Semantic talent search"
  longDescription: "Longer description."
classification:
  tags: ["llm"]
prerequisites:
  openshift:
    minimumVersion: "4.12"
deployment:
  supportedActions: ["CHECK_PRE_REQS", "STATUS", "INSTALL", "UNINSTALL_DELETE_ALL"]
  supportedModes: ["DEMO"]
  installer:
    image: "quay.io/rh-ai-quickstart/peoplemesh-installer:1.0.0"
    command: ["/installer/entrypoint.sh"]
    requiredEnv: ["ACTION", "TARGET_NAMESPACE", "INSTALL_MODE"]
  defaultNamespace: "peoplemesh-quickstart"
status:
  pollingInterval: "10s"
  timeout: "15m"
parameters:
  secrets:
    - name: "keycloak.realm.testUser.password"
      displayName: "Test User Password"
      description: "Password for the default test user"
      type: "password"
      required: true
      llmGuidance: "Always ask for this - never auto-generate."
  configuration:
    - name: "ollama.gpu.enabled"
      displayName: "Enable GPU for Ollama"
      type: "boolean"
      required: false
      default: false
llmContext:
  whenToRecommend: "Recommend for talent discovery."
"""
