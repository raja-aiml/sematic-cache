#!/usr/bin/env bash
set -eo pipefail

# ─────────────────────────────────────────────────────────────
# 🔧 CONFIGURATION
# ─────────────────────────────────────────────────────────────
IMAGE_NAME="sematic-cache:latest"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
DOCKERFILE="$REPO_ROOT/deploy/docker/Dockerfile"
APP_MANIFEST="$REPO_ROOT/deploy/k8s/app/sematic-cache.yaml"

CLUSTER_NAME="sematic-cache"
KUBE_CONTEXT="k3d-${CLUSTER_NAME}"
NAMESPACE_APP="app"

# ─────────────────────────────────────────────────────────────
# 📖 USAGE
# ─────────────────────────────────────────────────────────────
function usage() {
    cat <<EOM
Usage: $(basename "$0") <command>

Commands:
  build      Build and import Docker image into k3d cluster
  deploy     Apply app manifest to existing cluster
  test       Show app pod, service, and ingress status
EOM
}

# ─────────────────────────────────────────────────────────────
# 🔨 BUILD & IMPORT IMAGE
# ─────────────────────────────────────────────────────────────
function build_image() {
    echo "🔨 Building image '$IMAGE_NAME' using BuildKit..."
    DOCKER_BUILDKIT=1 docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" "$REPO_ROOT"

    echo "📦 Importing image into k3d cluster '$CLUSTER_NAME'..."
    k3d image import "$IMAGE_NAME" -c "$CLUSTER_NAME"
}

# ─────────────────────────────────────────────────────────────
# 🚀 DEPLOY APPLICATION ONLY
# ─────────────────────────────────────────────────────────────
function deploy_app() {
    echo "📁 Creating namespace '$NAMESPACE_APP' (if not exists)..."
    kubectl --context "$KUBE_CONTEXT" create namespace "$NAMESPACE_APP" --dry-run=client -o yaml | kubectl apply -f -

    echo "🚀 Applying app manifest: $APP_MANIFEST"
    kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" apply -f "$APP_MANIFEST"

    echo "⏳ Waiting for app deployment rollout..."
    kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" rollout status deployment/sematic-cache --timeout=60s
}

# ─────────────────────────────────────────────────────────────
# 🧪 TEST DEPLOYMENT
# ─────────────────────────────────────────────────────────────
function test_app() {
    echo "🔍 Testing Semantic Cache App..."
    kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get pods
    kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get svc
    kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get ingress
    echo "✅ App status check complete."
}

# ─────────────────────────────────────────────────────────────
# 🧭 MAIN
# ─────────────────────────────────────────────────────────────
function main() {
    case "$1" in
        build) build_image ;;
        deploy) deploy_app ;;
        test) test_app ;;
        help|*) usage ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [ $# -lt 1 ]; then usage; exit 1; fi
    main "$@"
fi