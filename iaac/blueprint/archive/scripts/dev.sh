#!/usr/bin/env bash
set -eo pipefail

# ─────────────────────────────────────────────────────────────
# 🔧 CONFIGURATION
# ─────────────────────────────────────────────────────────────
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
DEPLOY_DIR="$REPO_ROOT/deploy"
DOCKERFILE="$REPO_ROOT/Dockerfile"
APP_MANIFEST="$DEPLOY_DIR/app/"
ENV_FILE="$REPO_ROOT/.env"

CLUSTER_NAME="sematic-cache"
KUBE_CONTEXT="k3d-${CLUSTER_NAME}"
NAMESPACE_APP="app"
SECRET_NAME="sematic-cache-secrets"

# Use a simple, consistent image name
IMAGE_NAME="sematic-cache:working"

# ─────────────────────────────────────────────────────────────
# 🔐 LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────
function load_env_vars() {
    echo "🔐 Loading environment variables..."
    
    # Default values
    DATABASE_URL=${DATABASE_URL:-"postgres://cache:cache@postgres.infra:5432/cache?sslmode=disable"}
    OPENAI_API_KEY=${OPENAI_API_KEY:-"dummy-key"}
    
    # Load from .env file if it exists
    if [[ -f "$ENV_FILE" ]]; then
        echo "📄 Found .env file: $ENV_FILE"
        # Source the .env file, ignoring comments and empty lines
        set -a  # automatically export all variables
        source <(grep -v '^#' "$ENV_FILE" | grep -v '^$')
        set +a
        echo "✅ Environment variables loaded from .env"
    else
        echo "⚠️  No .env file found at $ENV_FILE, using default values"
    fi
    
    # Log the configuration (without sensitive values)
    echo "🔧 Configuration:"
    echo "   DATABASE_URL: ${DATABASE_URL//:*@/:***@}"  # Hide password
    echo "   OPENAI_API_KEY: ${OPENAI_API_KEY:0:8}..."   # Show only first 8 chars
}

# ─────────────────────────────────────────────────────────────
# 🔑 CREATE KUBERNETES SECRETS
# ─────────────────────────────────────────────────────────────
function create_secrets() {
    echo "🔑 Creating Kubernetes secrets..."
    
    # Check if secret already exists
    if kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get secret "$SECRET_NAME" >/dev/null 2>&1; then
        echo "🔄 Secret '$SECRET_NAME' already exists, updating..."
        kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" delete secret "$SECRET_NAME"
    fi
    
    # Create the secret
    kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" create secret generic "$SECRET_NAME" \
        --from-literal=DATABASE_URL="$DATABASE_URL" \
        --from-literal=OPENAI_API_KEY="$OPENAI_API_KEY"
    
    echo "✅ Secret '$SECRET_NAME' created successfully"
}

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
  remove     Delete app deployment and associated resources
  logs       Show application logs
  status     Show deployment status

Current image: $IMAGE_NAME
EOM
}

# ─────────────────────────────────────────────────────────────
# 🔨 BUILD & IMPORT IMAGE
# ─────────────────────────────────────────────────────────────
function build_image() {
    echo "🔨 Building image '$IMAGE_NAME'..."
    
    # Build with no cache to ensure fresh build
    docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" "$REPO_ROOT" --no-cache
    
    echo "🧪 Testing image locally..."
    if docker run --rm "$IMAGE_NAME" --help >/dev/null 2>&1; then
        echo "✅ Image test passed"
    else
        echo "❌ Image test failed - binary may not exist"
        echo "🔍 Checking image contents:"
        docker run --rm "$IMAGE_NAME" ls -la /app/ || echo "Cannot list /app/"
        return 1
    fi
    
    echo "📦 Importing image into k3d cluster '$CLUSTER_NAME'..."
    if k3d image import "$IMAGE_NAME" -c "$CLUSTER_NAME"; then
        echo "✅ Image imported successfully"
    else
        echo "❌ Failed to import image"
        return 1
    fi
    
    echo "🔍 Verifying image in k3d..."
    docker exec k3d-sematic-cache-server-0 crictl images | grep sematic-cache || echo "⚠️ Image not found in k3d"
}

# ─────────────────────────────────────────────────────────────
# 🚀 DEPLOY APPLICATION
# ─────────────────────────────────────────────────────────────
function deploy_app() {
    echo "🚀 Starting deployment process..."
    
    # Load environment variables first
    load_env_vars
    
    echo "📁 Creating namespace '$NAMESPACE_APP' (if not exists)..."
    kubectl --context "$KUBE_CONTEXT" create namespace "$NAMESPACE_APP" --dry-run=client -o yaml | kubectl apply -f -

    # Create secrets before deploying the app
    create_secrets

    echo "📦 Applying app manifest..."
    kubectl --context "$KUBE_CONTEXT" apply -k "$APP_MANIFEST"

    echo "⏳ Waiting for app deployment rollout..."
    if kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" rollout status deployment/sematic-cache --timeout=180s; then
        echo "✅ Deployment successful"
        show_status
    else
        echo "❌ Deployment failed or timed out"
        echo ""
        echo "🔍 Checking pod status..."
        kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get pods -o wide
        echo ""
        echo "📜 Pod describe (last pod):"
        POD=$(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get pods -l app=sematic-cache -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || echo "")
        if [[ -n "$POD" ]]; then
            kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" describe pod "$POD"
        fi
        echo ""
        echo "📜 Recent logs:"
        kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" logs deployment/sematic-cache --tail=20 || echo "No logs available"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────
# 📊 SHOW STATUS
# ─────────────────────────────────────────────────────────────
function show_status() {
    echo -e "\n📊 Application Status:"
    echo "🔹 Pods:"
    kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get pods -o wide
    
    echo -e "\n🔹 Services:"
    kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get svc
    
    echo -e "\n🔹 Ingress:"
    kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get ingress
    
    # Check if app is responding
    echo -e "\n🔹 Health Check:"
    POD=$(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get pods -l app=sematic-cache -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    if [[ -n "$POD" ]]; then
        if kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" exec "$POD" -- wget -qO- http://localhost:8080/health 2>/dev/null; then
            echo "✅ Application is responding"
        else
            echo "❌ Application not responding on /health endpoint"
        fi
    fi
    
    echo -e "\n🌐 Access URLs:"
    echo "   LoadBalancer: http://localhost:8080"
    echo "   Ingress:      http://sematic.127.0.0.1.nip.io:8080"
}

# ─────────────────────────────────────────────────────────────
# 🧪 TEST DEPLOYMENT
# ─────────────────────────────────────────────────────────────
function test_app() {
    echo "🔍 Testing Semantic Cache App..."
    show_status
    
    # Try to hit the health endpoint
    echo -e "\n🧪 Testing endpoints:"
    echo "Testing health endpoint..."
    if curl -s http://localhost:8080/health >/dev/null 2>&1; then
        echo "✅ Health endpoint accessible"
        curl -s http://localhost:8080/health | jq . 2>/dev/null || curl -s http://localhost:8080/health
    else
        echo "❌ Health endpoint not accessible"
        echo "💡 Try: kubectl port-forward -n app svc/sematic-cache 8080:8080"
    fi
}

# ─────────────────────────────────────────────────────────────
# 📜 SHOW LOGS
# ─────────────────────────────────────────────────────────────
function show_logs() {
    echo "📜 Application Logs:"
    kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" logs deployment/sematic-cache --tail=50 -f
}

# ─────────────────────────────────────────────────────────────
# 🗑 REMOVE DEPLOYMENT
# ─────────────────────────────────────────────────────────────
function remove_app() {
    echo "🗑 Deleting app resources from namespace '$NAMESPACE_APP'..."
    kubectl --context "$KUBE_CONTEXT" delete -k "$APP_MANIFEST" --ignore-not-found
    echo "✅ Removal complete."
}

# ─────────────────────────────────────────────────────────────
# 🧭 MAIN
# ─────────────────────────────────────────────────────────────
function main() {
    case "$1" in
        build) build_image ;;
        deploy) deploy_app ;;
        test) test_app ;;
        remove) remove_app ;;
        logs) show_logs ;;
        status) show_status ;;
        help|*) usage ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [ $# -lt 1 ]; then usage; exit 1; fi
    main "$@"
fi