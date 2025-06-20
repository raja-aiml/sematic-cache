#!/bin/bash
# composite-test.sh - Test composite backend with k3d cluster services
# This script sets up port forwarding and runs composite backend tests
# against services running in the k3d cluster

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSITE_CONFIG="${PROJECT_ROOT}/config/composite-cluster.yml"
COMPOSITE_NO_REDIS_CONFIG="${PROJECT_ROOT}/config/composite-cluster-no-redis.yml"
SERVER_PORT=8090
FORWARD_PIDS=""

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Stop port forwards
    if [[ -n "${FORWARD_PIDS}" ]]; then
        for pid in ${FORWARD_PIDS}; do
            if kill -0 "${pid}" 2>/dev/null; then
                kill "${pid}" 2>/dev/null || true
            fi
        done
    fi
    
    # Stop any running servers
    pkill -f "go run.*cmd/server/main.go" 2>/dev/null || true
    
    # Kill anything on the server port
    lsof -ti :${SERVER_PORT} | xargs kill -9 2>/dev/null || true
    
    # Remove temporary files
    rm -f "${PROJECT_ROOT}/server.log"
    
    log_success "Cleanup complete"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if cluster is running
    if ! k3d cluster list | grep -q "sematic-cache.*1/1.*0/0"; then
        log_error "k3d cluster 'sematic-cache' is not running"
        log_info "Run: ./deploy/cluster.sh up"
        exit 1
    fi
    
    # Check for kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check for Go
    if ! command -v go &> /dev/null; then
        log_error "go not found. Please install Go."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Get service namespace
get_service_namespace() {
    local namespace=""
    
    # Check infra namespace first
    if kubectl get svc -n infra postgres &> /dev/null; then
        namespace="infra"
    elif kubectl get svc -n semantic-cache postgres &> /dev/null; then
        namespace="semantic-cache"
    else
        log_error "PostgreSQL service not found in cluster"
        exit 1
    fi
    
    echo "${namespace}"
}

# Set up port forwarding
setup_port_forwarding() {
    local namespace=$1
    log_info "Setting up port forwarding from namespace: ${namespace}"
    
    # Port forward PostgreSQL
    log_info "Port forwarding PostgreSQL (5432)..."
    kubectl port-forward -n "${namespace}" svc/postgres 5432:5432 > /dev/null 2>&1 &
    local pg_pid=$!
    FORWARD_PIDS="${FORWARD_PIDS} ${pg_pid}"
    
    # Port forward Redis
    log_info "Port forwarding Redis (6379)..."
    kubectl port-forward -n "${namespace}" svc/redis 6379:6379 > /dev/null 2>&1 &
    local redis_pid=$!
    FORWARD_PIDS="${FORWARD_PIDS} ${redis_pid}"
    
    # Wait for port forwards to be ready
    sleep 3
    
    # Test connections
    if nc -zv localhost 5432 &> /dev/null; then
        log_success "PostgreSQL port forward established"
    else
        log_error "PostgreSQL port forward failed"
        exit 1
    fi
    
    if nc -zv localhost 6379 &> /dev/null; then
        log_success "Redis port forward established"
    else
        log_error "Redis port forward failed"
        exit 1
    fi
}

# Get database credentials
get_db_credentials() {
    local namespace=$1
    local pg_pod=$(kubectl get pod -n "${namespace}" -l app=postgres -o jsonpath='{.items[0].metadata.name}')
    
    # Get environment variables from pod
    local db_user=$(kubectl exec -n "${namespace}" "${pg_pod}" -- printenv POSTGRES_USER 2>/dev/null || echo "postgres")
    local db_pass=$(kubectl exec -n "${namespace}" "${pg_pod}" -- printenv POSTGRES_PASSWORD 2>/dev/null || echo "postgres")
    local db_name=$(kubectl exec -n "${namespace}" "${pg_pod}" -- printenv POSTGRES_DB 2>/dev/null || echo "cache")
    
    echo "postgres://${db_user}:${db_pass}@localhost:5432/${db_name}?sslmode=disable"
}

# Initialize database
init_database() {
    local namespace=$1
    local pg_pod=$(kubectl get pod -n "${namespace}" -l app=postgres -o jsonpath='{.items[0].metadata.name}')
    
    log_info "Initializing database..."
    
    # Get credentials
    local db_user=$(kubectl exec -n "${namespace}" "${pg_pod}" -- printenv POSTGRES_USER 2>/dev/null || echo "postgres")
    local db_name=$(kubectl exec -n "${namespace}" "${pg_pod}" -- printenv POSTGRES_DB 2>/dev/null || echo "cache")
    
    # Create database if not exists
    kubectl exec -n "${namespace}" "${pg_pod}" -- psql -U "${db_user}" -tc "SELECT 1 FROM pg_database WHERE datname = '${db_name}'" | grep -q 1 || \
        kubectl exec -n "${namespace}" "${pg_pod}" -- psql -U "${db_user}" -c "CREATE DATABASE ${db_name};"
    
    # Enable pgvector extension
    kubectl exec -n "${namespace}" "${pg_pod}" -- psql -U "${db_user}" -d "${db_name}" -c "CREATE EXTENSION IF NOT EXISTS vector;" || true
    
    log_success "Database initialized"
}

# Create composite configuration
create_composite_config() {
    log_info "Creating composite configuration..."
    
    # Create config without Redis (due to cluster mode issue)
    cat > "${COMPOSITE_NO_REDIS_CONFIG}" <<EOF
# Composite Backend Configuration for Cluster Services
# Uses Memory and PostgreSQL tiers (Redis excluded due to cluster mode)

server:
  address: ":${SERVER_PORT}"
  
cache:
  type: "composite"
  min_similarity: 0.85
  
  composite:
    promote_on_hit: true
    
    tiers:
      - name: "memory-l1"
        type: "memory"
        priority: 1
        capacity: 1000
        eviction_policy: "LRU"
        
      - name: "postgres-l2"
        type: "gorm"
        priority: 2
        # Uses DATABASE_URL environment variable

openai:
  api_key: ""  # Leave empty to use OPENAI_API_KEY environment variable
  base_url: "https://api.openai.com/v1"
EOF
    
    log_success "Configuration created: ${COMPOSITE_NO_REDIS_CONFIG}"
}

# Run composite demo
run_composite_demo() {
    local db_url=$1
    
    log_info "Running composite backend demo..."
    
    cd "${PROJECT_ROOT}/examples/composite-demo"
    
    export DATABASE_URL="${db_url}"
    
    # Run with timeout
    timeout 30s go run main.go -config "${COMPOSITE_NO_REDIS_CONFIG}" || {
        if [ $? -eq 124 ]; then
            log_warning "Demo timed out (this is normal)"
        else
            log_error "Demo failed"
            return 1
        fi
    }
    
    log_success "Demo completed"
}

# Run server test
run_server_test() {
    local db_url=$1
    
    log_info "Starting semantic cache server on port ${SERVER_PORT}..."
    
    cd "${PROJECT_ROOT}"
    
    export DATABASE_URL="${db_url}"
    
    # Kill any existing process on the port first
    lsof -ti :${SERVER_PORT} | xargs kill -9 2>/dev/null || true
    sleep 1
    
    # Start server in background
    go run cmd/server/main.go -config "${COMPOSITE_NO_REDIS_CONFIG}" > server.log 2>&1 &
    local server_pid=$!
    
    # Wait for server to start
    sleep 3
    
    # Check if server is running
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        log_error "Server failed to start. Check server.log for details:"
        tail -20 server.log
        return 1
    fi
    
    log_success "Server started on port ${SERVER_PORT}"
    
    # Test endpoints
    log_info "Testing server endpoints..."
    
    # Health check
    if curl -s "http://localhost:${SERVER_PORT}/health" | grep -q "OK"; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        return 1
    fi
    
    # Set a cache entry
    curl -X POST "http://localhost:${SERVER_PORT}/set" \
        -H "Content-Type: application/json" \
        -d '{"prompt":"test composite backend","answer":"working perfectly!"}' \
        -s --max-time 5 > /dev/null
    
    # Get the cache entry
    local response=$(curl -X POST "http://localhost:${SERVER_PORT}/get" \
        -H "Content-Type: application/json" \
        -d '{"prompt":"test composite backend"}' \
        -s --max-time 5)
    
    if echo "${response}" | grep -q "working perfectly"; then
        log_success "Cache operations working"
    else
        log_error "Cache operations failed"
        return 1
    fi
    
    # Check metrics
    local metrics=$(curl -s --max-time 5 "http://localhost:${SERVER_PORT}/metrics")
    echo -e "${BLUE}Cache Metrics:${NC} ${metrics}"
    
    # Stop server
    kill "${server_pid}" 2>/dev/null || true
    
    log_success "Server tests completed"
}

# Main function
main() {
    local cmd=${1:-"test"}
    
    case "${cmd}" in
        test)
            log_info "Starting composite backend test with cluster services"
            
            # Check prerequisites
            check_prerequisites
            
            # Get service namespace
            local namespace=$(get_service_namespace)
            log_info "Using namespace: ${namespace}"
            
            # Set up port forwarding
            setup_port_forwarding "${namespace}"
            
            # Get database credentials
            local db_url=$(get_db_credentials "${namespace}")
            log_info "Database URL: ${db_url}"
            
            # Initialize database
            init_database "${namespace}"
            
            # Create configuration
            create_composite_config
            
            # Check for OpenAI API key
            if [[ -z "${OPENAI_API_KEY:-}" ]]; then
                log_warning "OPENAI_API_KEY not set. Embedding generation will fail, but exact match caching will work."
            fi
            
            # Run tests
            log_info "Running composite backend tests..."
            
            # Run demo
            run_composite_demo "${db_url}"
            
            echo ""
            
            # Run server test
            run_server_test "${db_url}"
            
            log_success "All tests completed successfully!"
            ;;
            
        demo)
            # Just run the demo
            check_prerequisites
            local namespace=$(get_service_namespace)
            setup_port_forwarding "${namespace}"
            local db_url=$(get_db_credentials "${namespace}")
            init_database "${namespace}"
            create_composite_config
            
            export DATABASE_URL="${db_url}"
            export OPENAI_API_KEY="${OPENAI_API_KEY:-your-api-key}"
            
            log_info "Running composite demo..."
            cd "${PROJECT_ROOT}/examples/composite-demo"
            go run main.go -config "${COMPOSITE_NO_REDIS_CONFIG}"
            ;;
            
        server)
            # Just run the server
            check_prerequisites
            local namespace=$(get_service_namespace)
            setup_port_forwarding "${namespace}"
            local db_url=$(get_db_credentials "${namespace}")
            init_database "${namespace}"
            create_composite_config
            
            export DATABASE_URL="${db_url}"
            
            log_info "Starting server on port ${SERVER_PORT}..."
            log_info "Access the server at: http://localhost:${SERVER_PORT}"
            log_info "Press Ctrl+C to stop"
            
            cd "${PROJECT_ROOT}"
            go run cmd/server/main.go -config "${COMPOSITE_NO_REDIS_CONFIG}"
            ;;
            
        port-forward)
            # Just set up port forwarding
            check_prerequisites
            local namespace=$(get_service_namespace)
            setup_port_forwarding "${namespace}"
            
            log_success "Port forwarding established"
            log_info "PostgreSQL: localhost:5432"
            log_info "Redis: localhost:6379"
            log_info "Press Ctrl+C to stop"
            
            # Keep running
            while true; do
                sleep 1
            done
            ;;
            
        help|--help|-h)
            cat <<EOF
composite-test.sh - Test composite backend with k3d cluster services

Usage: $0 [command]

Commands:
    test          Run all composite backend tests (default)
    demo          Run only the composite demo
    server        Start the server with composite backend
    port-forward  Set up port forwarding only
    help          Show this help message

Prerequisites:
    - k3d cluster 'sematic-cache' must be running
    - kubectl must be installed
    - Go must be installed

Examples:
    # Run all tests
    $0 test
    
    # Run demo with custom API key
    OPENAI_API_KEY="your-key" $0 demo
    
    # Start server for development
    $0 server
    
    # Set up port forwarding for manual testing
    $0 port-forward
EOF
            ;;
            
        *)
            log_error "Unknown command: ${cmd}"
            echo "Run '$0 help' for usage"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"