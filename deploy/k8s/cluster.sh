#!/usr/bin/env bash
set -e

CLUSTER_NAME="sematic-cache"
KUBE_CONTEXT="k3d-${CLUSTER_NAME}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${ROOT_DIR}/config"
INFRA_DIR="${ROOT_DIR}/infra"
APP_DIR="${ROOT_DIR}/app"

usage() {
    cat <<EOM
Usage: $(basename "$0") <command> [args]

Commands:
  up                    Create k3d cluster and apply manifests
  down                  Delete k3d cluster
  ps                    List k3d clusters and Kubernetes resources
  logs [POD] [--follow] Show logs for a pod (default: all pods). Use --follow to tail logs.
  help                  Show this help message
EOM
}

cmd_up() {
    echo "🚀 Creating k3d cluster '$CLUSTER_NAME'..."
    k3d cluster create "$CLUSTER_NAME" \
        --agents 0 \
        --port "8080:8080@loadbalancer" \
        --registry-config "${CONFIG_DIR}/k3d-registry.yaml" \
        --kubeconfig-switch-context

    echo "⏳ Waiting for cluster to be ready..."
    kubectl config use-context "$KUBE_CONTEXT"
    kubectl --context "$KUBE_CONTEXT" wait --for=condition=Ready node/"${KUBE_CONTEXT}-server-0" --timeout=60s

    echo "📦 Applying infrastructure manifests..."
    for file in "${INFRA_DIR}"/*.yaml; do
        echo "↪ applying: $file"
        kubectl --context "$KUBE_CONTEXT" apply -f "$file"
    done

    echo "📦 Deploying application..."
    kubectl --context "$KUBE_CONTEXT" apply -f "${APP_DIR}/sematic-cache.yaml"

    echo "✅ Waiting for deployments to become ready..."
    kubectl --context "$KUBE_CONTEXT" rollout status deployment/postgres --timeout=120s || true
    kubectl --context "$KUBE_CONTEXT" rollout status deployment/redis --timeout=120s || true
    kubectl --context "$KUBE_CONTEXT" rollout status deployment/sematic-cache --timeout=120s || true
}

cmd_down() {
    echo "🗑️  Deleting k3d cluster '$CLUSTER_NAME'..."
    k3d cluster delete "$CLUSTER_NAME"
}

cmd_ps() {
    echo "📋 k3d clusters:"
    k3d cluster list
    echo
    echo "📋 Kubernetes resources in context '$KUBE_CONTEXT':"
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

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

case "$1" in
    up) cmd_up ;;
    down) cmd_down ;;
    ps) cmd_ps ;;
    logs) shift; cmd_logs "$@" ;;
    help|*) usage ;;
esac