#!/usr/bin/env bash
set -eo pipefail

# ─────────────────────────────────────────────────────────────
# 🔧 CONFIGURATION
# ─────────────────────────────────────────────────────────────
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
COMPOSE_FILE="$REPO_ROOT/deploy/docker/docker-compose.yml"
ENV_FILE="$REPO_ROOT/.env"

PROJECT_NAME="sematic-cache"

# ─────────────────────────────────────────────────────────────
# 🔐 LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────
function load_env_vars() {
    echo "🔐 Loading environment variables..."
    
    # Default values
    export DATABASE_URL=${DATABASE_URL:-"postgres://cache:cache@postgres:5432/cache?sslmode=disable"}
    export OPENAI_API_KEY=${OPENAI_API_KEY:-"dummy-key"}
    export JAEGER_ENDPOINT=${JAEGER_ENDPOINT:-""}
    
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
    echo "   JAEGER_ENDPOINT: ${JAEGER_ENDPOINT:-"(not set)"}"
}

# ─────────────────────────────────────────────────────────────
# 📖 USAGE
# ─────────────────────────────────────────────────────────────
function usage() {
    cat <<EOM
Usage: $(basename "$0") <command>

Commands:
  up         Build and start all services
  down       Stop and remove all services
  build      Build application image
  logs       Show logs from all services
  status     Show status of all services
  test       Test the deployment endpoints
  clean      Remove all containers, networks, and volumes

Environment:
  Uses .env file in project root for configuration
  Falls back to defaults if .env file not found

Access URLs after deployment:
  Web Interface:    http://localhost:8080/web/
  API Health:       http://localhost:8080/semantic-cache/health
  API Metrics:      http://localhost:8080/semantic-cache/metrics
EOM
}

# ─────────────────────────────────────────────────────────────
# 🚀 START SERVICES
# ─────────────────────────────────────────────────────────────
function start_services() {
    echo "🚀 Starting Docker Compose deployment..."
    
    # Load environment variables first
    load_env_vars
    
    echo "🔨 Building and starting services..."
    COMPOSE_BAKE=true docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up --build -d
    
    echo "⏳ Waiting for services to be ready..."
    sleep 5
    
    # Wait for app to be healthy
    echo "🔍 Checking application health..."
    local retries=30
    while [[ $retries -gt 0 ]]; do
        if curl -s http://localhost:8080/semantic-cache/health >/dev/null 2>&1; then
            echo "✅ Application is ready!"
            show_status
            return 0
        fi
        echo "⏳ Waiting for application... ($retries retries left)"
        sleep 2
        ((retries--))
    done
    
    echo "❌ Application failed to become ready"
    echo "📜 Recent logs:"
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs app --tail=20
    return 1
}

# ─────────────────────────────────────────────────────────────
# 🛑 STOP SERVICES
# ─────────────────────────────────────────────────────────────
function stop_services() {
    echo "🛑 Stopping Docker Compose services..."
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down
    echo "✅ Services stopped"
}

# ─────────────────────────────────────────────────────────────
# 🔨 BUILD APPLICATION
# ─────────────────────────────────────────────────────────────
function build_app() {
    echo "🔨 Building application image..."
    load_env_vars
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" build app
    echo "✅ Application image built"
}

# ─────────────────────────────────────────────────────────────
# 📜 SHOW LOGS
# ─────────────────────────────────────────────────────────────
function show_logs() {
    echo "📜 Application Logs:"
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f
}

# ─────────────────────────────────────────────────────────────
# 📊 SHOW STATUS
# ─────────────────────────────────────────────────────────────
function show_status() {
    echo -e "\n📊 Service Status:"
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps
    
    echo -e "\n🌐 Access URLs:"
    echo "   Web Interface:    http://localhost:8080/web/"
    echo "   API Health:       http://localhost:8080/semantic-cache/health"
    echo "   API Metrics:      http://localhost:8080/semantic-cache/metrics"
    echo "   Proxy Health:     http://localhost:8080/proxy-health"
    
    echo -e "\n🔍 Quick Health Check:"
    if curl -s http://localhost:8080/semantic-cache/health >/dev/null 2>&1; then
        echo "   ✅ Semantic Cache API: Ready"
    else
        echo "   ❌ Semantic Cache API: Not responding"
    fi
    
    if curl -s http://localhost:8080/web/ >/dev/null 2>&1; then
        echo "   ✅ Web Interface: Ready"
    else
        echo "   ❌ Web Interface: Not responding"
    fi
}

# ─────────────────────────────────────────────────────────────
# 🧪 TEST DEPLOYMENT
# ─────────────────────────────────────────────────────────────
function test_deployment() {
    echo "🧪 Testing Semantic Cache Docker Deployment..."
    
    echo -e "\n🔍 Testing endpoints:"
    
    # Test health endpoint
    echo "Testing health endpoint..."
    if response=$(curl -s http://localhost:8080/semantic-cache/health); then
        echo "✅ Health endpoint: $response"
    else
        echo "❌ Health endpoint failed"
    fi
    
    # Test metrics endpoint
    echo "Testing metrics endpoint..."
    if response=$(curl -s http://localhost:8080/semantic-cache/metrics); then
        echo "✅ Metrics endpoint: $response"
    else
        echo "❌ Metrics endpoint failed"
    fi
    
    # Test web interface
    echo "Testing web interface..."
    if curl -s http://localhost:8080/web/ | grep -q "Semantic Cache"; then
        echo "✅ Web interface: Available"
    else
        echo "❌ Web interface failed"
    fi
    
    # Test API functionality
    echo "Testing API set/get functionality..."
    if curl -s -X POST -H "Content-Type: application/json" \
        -d '{"prompt": "Docker test", "answer": "Docker deployment working", "modelName": "test"}' \
        http://localhost:8080/semantic-cache/set >/dev/null 2>&1; then
        
        if response=$(curl -s -X POST -H "Content-Type: application/json" \
            -d '{"prompt": "Docker test"}' \
            http://localhost:8080/semantic-cache/get); then
            echo "✅ API set/get: $response"
        else
            echo "❌ API get failed"
        fi
    else
        echo "❌ API set failed"
    fi
    
    show_status
}

# ─────────────────────────────────────────────────────────────
# 🧹 CLEAN UP
# ─────────────────────────────────────────────────────────────
function clean_up() {
    echo "🧹 Cleaning up Docker resources..."
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down -v
    docker system prune -f
    echo "✅ Cleanup complete"
}

# ─────────────────────────────────────────────────────────────
# 🧭 MAIN
# ─────────────────────────────────────────────────────────────
function main() {
    case "$1" in
        up) start_services ;;
        down) stop_services ;;
        build) build_app ;;
        logs) show_logs ;;
        status) show_status ;;
        test) test_deployment ;;
        clean) clean_up ;;
        help|*) usage ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [ $# -lt 1 ]; then usage; exit 1; fi
    main "$@"
fi