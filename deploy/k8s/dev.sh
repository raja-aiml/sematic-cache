#!/usr/bin/env bash
set -e

IMAGE_NAME="sematic-cache:latest"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CLUSTER_SCRIPT="$(cd "$(dirname "$0")" && pwd)/cluster.sh"

usage() {
    cat <<EOM
Usage: $(basename "$0") <command>

Commands:
  build     Build Docker image and push to localhost:5000
  deploy    Build, push, and deploy to k3d cluster
  status    Show k3d cluster and Kubernetes resources
EOM
}

build_image() {
    echo "🔨 Building image '$IMAGE_NAME'..."
    docker build -t "$IMAGE_NAME" -f "$REPO_ROOT/deploy/docker/Dockerfile" "$REPO_ROOT"
    docker tag "$IMAGE_NAME" "localhost:5000/$IMAGE_NAME"
    docker push "localhost:5000/$IMAGE_NAME"
}

cmd_build() {
    build_image
}

cmd_deploy() {
    build_image
    "$CLUSTER_SCRIPT" up
}

cmd_status() {
    "$CLUSTER_SCRIPT" ps
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

case "$1" in
    build) cmd_build ;;
    deploy) cmd_deploy ;;
    status) cmd_status ;;
    help|*) usage ;;
esac