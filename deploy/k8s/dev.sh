#!/usr/bin/env bash

set -e

REGISTRY_NAME="sematic-registry"
IMAGE_NAME="sematic-cache:latest"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CLUSTER_SCRIPT="$(cd "$(dirname "$0")" && pwd)/cluster.sh"
CLUSTER_NAME="sematic-cache"
KUBE_CONTEXT="k3d-${CLUSTER_NAME}"
APP_MANIFEST="$REPO_ROOT/deploy/k8s/sematic-cache.yaml"

usage() {
    cat <<EOM
Usage: $(basename "$0") <command>

Commands:
  build    Build the Docker image and push to local registry
  deploy   Build image, create cluster, and deploy application
  down     Tear down cluster and stop registry
  help     Show this help message
EOM
}

ensure_registry() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${REGISTRY_NAME}$"; then
        echo "Starting local Docker registry '${REGISTRY_NAME}'..."
        docker run -d -p 5000:5000 --name "$REGISTRY_NAME" registry:2
    else
        echo "Local Docker registry '${REGISTRY_NAME}' already running."
    fi
}

cleanup_registry() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${REGISTRY_NAME}$"; then
        echo "Deleting local Docker registry '${REGISTRY_NAME}'..."
        docker rm -f "$REGISTRY_NAME" >/dev/null
    fi
}

build_image() {
    docker build -t "$IMAGE_NAME" -f "$REPO_ROOT/deploy/docker/Dockerfile" "$REPO_ROOT"
}

push_image() {
    ensure_registry
    docker tag "$IMAGE_NAME" "localhost:5000/$IMAGE_NAME"
    docker push "localhost:5000/$IMAGE_NAME"
}

deploy_app() {
    kubectl --context "$KUBE_CONTEXT" apply -f "$APP_MANIFEST"
    kubectl --context "$KUBE_CONTEXT" rollout status deployment/sematic-cache --timeout=120s
}

cmd_build() {
    build_image
    push_image
}

cmd_deploy() {
    cmd_build
    "$CLUSTER_SCRIPT" up
    deploy_app
}

cmd_down() {
    "$CLUSTER_SCRIPT" down
    cleanup_registry
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

case "$1" in
    build)
        cmd_build
        ;;
    deploy)
        cmd_deploy
        ;;
    down)
        cmd_down
        ;;
    help|*)
        usage
        ;;
esac
