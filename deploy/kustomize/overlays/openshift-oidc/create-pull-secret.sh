#!/usr/bin/env bash
# WARNING: This script is for development purposes only and is not intended for production use.
set -euo pipefail

NAMESPACE="${1:-rhoai-mcp}"
SECRET_NAME="ghcr-pull-secret"
DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker/config.json}"

if [ ! -f "$DOCKER_CONFIG" ]; then
  echo "Error: $DOCKER_CONFIG not found" >&2
  exit 1
fi

ghcr_auth=$(jq -r '.auths["ghcr.io"] // empty' "$DOCKER_CONFIG")
if [ -z "$ghcr_auth" ]; then
  echo "Error: no ghcr.io entry found in $DOCKER_CONFIG" >&2
  exit 1
fi

dockerconfigjson=$(jq -n --argjson auth "$ghcr_auth" '{"auths":{"ghcr.io": $auth}}')

kubectl create secret generic "$SECRET_NAME" \
  --namespace="$NAMESPACE" \
  --type=kubernetes.io/dockerconfigjson \
  --from-literal=".dockerconfigjson=$dockerconfigjson" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Secret '$SECRET_NAME' created in namespace '$NAMESPACE'"
