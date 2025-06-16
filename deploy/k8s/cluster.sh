#!/usr/bin/env bash
set -e

CLUSTER_NAME="sematic-cache"
KUBE_CONTEXT="k3d-${CLUSTER_NAME}"
NAMESPACE="infra"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="${ROOT_DIR}/infra"

function usage() {
    cat <<EOM
Usage: $(basename "$0") <command> [args]

Commands:
  up                    Create k3d cluster and apply manifests
  down                  Delete k3d cluster
  ps                    List k3d clusters and Kubernetes resources
  logs [POD] [--follow] Show logs for a pod (default: all pods). Use --follow to tail logs.
  test                  Run basic connectivity & deployment checks
  help                  Show this help message
EOM
}

function cmd_up() {
    echo "🚀 Creating k3d cluster '$CLUSTER_NAME'..."
    k3d cluster create "$CLUSTER_NAME" \
        --agents 0 \
        --port "8080:80@loadbalancer" \
        --port "8443:443@loadbalancer" \
        --k3s-arg "--disable=traefik@server:0" \
        --kubeconfig-switch-context

    echo "⏳ Waiting for cluster to be ready..."
    kubectl config use-context "$KUBE_CONTEXT"
    kubectl --context "$KUBE_CONTEXT" wait --for=condition=Ready node/"${KUBE_CONTEXT}-server-0" --timeout=60s

    echo "📁 Creating namespace: $NAMESPACE"
    kubectl --context "$KUBE_CONTEXT" create namespace "$NAMESPACE" || true

    echo "📦 Applying infrastructure manifests (via Kustomize)..."
    kubectl --context "$KUBE_CONTEXT" apply -k "$INFRA_DIR"

    echo "🌐 Waiting for ingress-nginx controller to be ready..."
    kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" rollout status deployment/ingress-nginx-controller --timeout=120s || true

    echo "🔐 Waiting for ingress-nginx admission webhooks to be ready..."
    for job in ingress-nginx-admission-create ingress-nginx-admission-patch; do
        echo "⏳ Waiting for job $job..."
        kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" wait --for=condition=complete job/$job --timeout=60s || {
            echo "❌ Job $job failed or timed out"; exit 1;
        }
    done

    echo "✅ Waiting for other deployments to become ready..."
    for d in postgres redis ; do
        kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" rollout status deployment/$d --timeout=120s || true
    done
}

function cmd_down() {
    echo "🗑️  Deleting k3d cluster '$CLUSTER_NAME'..."
    k3d cluster delete "$CLUSTER_NAME"
}

function cmd_ps() {
    echo "📋 k3d clusters:"
    k3d cluster list
    echo
    echo "📋 Kubernetes resources in namespace '$NAMESPACE':"
    kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" get all
}

function cmd_logs() {
    local pod="$1"
    shift || true
    if [ -z "$pod" ]; then
        pods=$(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" get pods -o name)
        for p in $pods; do
            echo "=== Logs for $p ==="
            kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" logs $p "$@"
            echo
        done
    else
        kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" logs "$pod" "$@"
    fi
}

function cmd_test() {
    echo -e "\n🔍 Verifying Kubernetes deployments:"
    for d in postgres redis ; do
        echo -n "↪ $d: "
        kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" get deployment "$d" -o=jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "❌ Missing"
        echo
    done
}

function main() {
    if [ $# -lt 1 ]; then
        usage
        exit 1
    fi

    case "$1" in
        up) cmd_up ;;
        down) cmd_down ;;
        ps) cmd_ps ;;
        logs) shift; cmd_logs "$@" ;;
        test) cmd_test ;;
        help|*) usage ;;
    esac
}

# Entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi