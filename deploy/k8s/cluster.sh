#!/usr/bin/env bash

set -e

# Name of the k3d cluster
CLUSTER_NAME="sematic-cache"
# Kubernetes context name for this k3d cluster
KUBE_CONTEXT="k3d-${CLUSTER_NAME}"

# Directory containing Kubernetes manifests
MANIFEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repository root (for building Docker image)
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$MANIFEST_DIR/../..")"
DOCKERFILE="$REPO_ROOT/deploy/docker/Dockerfile"

usage() {
    cat <<EOM
Usage: $(basename "$0") <command> [args]

Commands:
  up                    Create k3d cluster and deploy Kubernetes resources
  down                  Delete k3d cluster
  ps                    List k3d clusters and Kubernetes resources
  logs [POD] [--follow] Show logs for a pod (default: all pods). Use --follow to tail logs.
  help                  Show this help message
EOM
}

cmd_up() {
    echo "Creating k3d cluster '$CLUSTER_NAME'..."
    # Create cluster and switch kubectl context to it
    k3d cluster create "$CLUSTER_NAME" --agents 0 --port "8080:8080@loadbalancer" --kubeconfig-switch-context
    echo "Switching kubectl context to '$KUBE_CONTEXT'..."
    kubectl config use-context "$KUBE_CONTEXT"
    echo "Building local server image 'sematic-cache:latest'..."
    docker build -t sematic-cache:latest -f "$DOCKERFILE" "$REPO_ROOT"
    echo "Importing 'sematic-cache:latest' into k3d cluster '$CLUSTER_NAME'..."
    k3d image import sematic-cache:latest -c "$CLUSTER_NAME"
    echo "Waiting for Kubernetes nodes to be ready..."
    # Wait for the k3d control-plane node (named <context>-server-0)
    kubectl --context "$KUBE_CONTEXT" wait --for=condition=Ready node/"${KUBE_CONTEXT}-server-0" --timeout=60s
    echo "Applying Kubernetes manifests..."
    kubectl --context "$KUBE_CONTEXT" apply -f "$MANIFEST_DIR"
    echo "Waiting for deployments to be ready..."
    kubectl --context "$KUBE_CONTEXT" rollout status deployment/postgres --timeout=120s
    kubectl --context "$KUBE_CONTEXT" rollout status deployment/redis --timeout=120s
    kubectl --context "$KUBE_CONTEXT" rollout status deployment/sematic-cache --timeout=120s
    echo "Cluster is up and running. Application is available at http://localhost:8080"
}

cmd_down() {
    echo "Deleting k3d cluster '$CLUSTER_NAME'..."
    k3d cluster delete "$CLUSTER_NAME"
}

cmd_ps() {
    echo "k3d clusters:"
    k3d cluster list
    echo
    echo "Kubernetes resources in context '$KUBE_CONTEXT':"
    kubectl --context "$KUBE_CONTEXT" get all
}

cmd_logs() {
    local pod="$1"
    shift || true
    if [ -z "$pod" ]; then
        pods=$(kubectl --context "$KUBE_CONTEXT" get pods -o name)
        for p in $pods; do
            echo "=== Logs for $p ==="
            kubectl --context "$KUBE_CONTEXT" logs $p "$@"
            echo
        done
    else
        kubectl --context "$KUBE_CONTEXT" logs "$pod" "$@"
    fi
}

# Main
if [ $# -lt 1 ]; then
    usage
    exit 1
fi

COMMAND="$1"
shift

case "$COMMAND" in
    up)
        cmd_up
        ;;
    down)
        cmd_down
        ;;
    ps)
        cmd_ps
        ;;
    logs)
        cmd_logs "$@"
        ;;
    help|*)
        usage
        ;;
esac
