#!/usr/bin/env bash
# =====================================
# 🚀 PRODUCTION-READY E2E WORKFLOW
# =====================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_LOG="${SCRIPT_DIR}/workflow.log"
CLUSTER_NAME="sematic-cache"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[$timestamp]${NC} $message" | tee -a "$WORKFLOW_LOG"
}

success() {
    echo -e "${GREEN}✅ $1${NC}" | tee -a "$WORKFLOW_LOG"
}

error() {
    echo -e "${RED}❌ $1${NC}" | tee -a "$WORKFLOW_LOG"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" | tee -a "$WORKFLOW_LOG"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error "Workflow failed with exit code $exit_code"
        log "Check the log file: $WORKFLOW_LOG"
    fi
    exit $exit_code
}

trap cleanup EXIT

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [COMMAND]

Production-ready end-to-end workflow for semantic cache deployment.

COMMANDS:
  full               Complete workflow: cluster up → build → deploy → test → cleanup
  setup              Create cluster and deploy infrastructure  
  build              Build application image
  deploy             Deploy application to cluster
  test               Run comprehensive tests
  quick-test         Run essential tests only
  cleanup            Destroy cluster and cleanup resources
  status             Show current deployment status
  logs               Show application logs
  reset              Complete reset: cleanup → setup → build → deploy

OPTIONS:
  -h, --help         Show this help message
  -v, --verbose      Enable verbose output
  --skip-cleanup     Don't cleanup cluster after full workflow
  --timeout N        Set test timeout (default: 600s)
  --quick           Use quick mode for faster execution

EXAMPLES:
  $(basename "$0") full                    # Complete end-to-end workflow
  $(basename "$0") --quick full            # Quick workflow for development
  $(basename "$0") setup build deploy     # Step-by-step deployment
  $(basename "$0") test                    # Test existing deployment
  $(basename "$0") reset                   # Complete reset and redeploy

WORKFLOW STEPS:
  1. 🏗️  Setup:     Create k3d cluster and deploy infrastructure
  2. 🔨 Build:      Build Docker image and import to cluster  
  3. 🚀 Deploy:     Deploy application with secrets and configuration
  4. 🧪 Test:       Run comprehensive end-to-end tests
  5. 📊 Report:     Generate test report and status summary
  6. 🧹 Cleanup:    Optional cluster cleanup

EOF
}

check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    local missing=0
    for cmd in kubectl k3d docker curl; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            error "Required command '$cmd' not found"
            ((missing++))
        fi
    done
    
    if [ $missing -gt 0 ]; then
        error "$missing required commands are missing"
        log "Please install missing prerequisites and try again"
        return 1
    fi
    
    success "All prerequisites are available"
    return 0
}

step_setup() {
    log "🏗️  Step 1: Setting up cluster and infrastructure..."
    
    # Check if cluster already exists
    if k3d cluster list | grep -q "$CLUSTER_NAME"; then
        warning "Cluster '$CLUSTER_NAME' already exists"
        log "Checking cluster health..."
        
        if kubectl cluster-info >/dev/null 2>&1; then
            success "Existing cluster is healthy"
            return 0
        else
            warning "Existing cluster appears unhealthy, recreating..."
            "$SCRIPT_DIR/cluster.sh" down
        fi
    fi
    
    # Create cluster
    if "$SCRIPT_DIR/cluster.sh" up; then
        success "Cluster and infrastructure setup completed"
        
        # Verify infrastructure
        log "Verifying infrastructure components..."
        if "$SCRIPT_DIR/cluster.sh" test; then
            success "Infrastructure verification passed"
        else
            error "Infrastructure verification failed"
            return 1
        fi
    else
        error "Failed to setup cluster and infrastructure"
        return 1
    fi
}

step_build() {
    log "🔨 Step 2: Building application..."
    
    # Check if Dockerfile exists
    if [ ! -f "$SCRIPT_DIR/../Dockerfile" ]; then
        error "Dockerfile not found at project root"
        return 1
    fi
    
    # Build application
    if "$SCRIPT_DIR/dev.sh" build; then
        success "Application build completed"
    else
        error "Application build failed"
        return 1
    fi
}

step_deploy() {
    log "🚀 Step 3: Deploying application..."
    
    # Deploy application
    if "$SCRIPT_DIR/dev.sh" deploy; then
        success "Application deployment completed"
        
        # Wait for deployment to be ready
        log "Waiting for deployment to stabilize..."
        sleep 30
        
        # Quick deployment verification
        if "$SCRIPT_DIR/dev.sh" status >/dev/null 2>&1; then
            success "Application deployment verified"
        else
            warning "Application deployment verification inconclusive"
        fi
    else
        error "Application deployment failed"
        return 1
    fi
}

step_test() {
    log "🧪 Step 4: Running comprehensive tests..."
    
    local test_mode="all"
    if [ "${QUICK_MODE:-false}" = "true" ]; then
        test_mode="--quick all"
        log "Running quick test suite..."
    else
        log "Running comprehensive test suite..."
    fi
    
    # Run tests with timeout
    if timeout "${TEST_TIMEOUT:-600}" "$SCRIPT_DIR/tests/e2e.sh" $test_mode; then
        success "All tests passed successfully"
        generate_test_report
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            error "Tests timed out after ${TEST_TIMEOUT:-600} seconds"
        else
            error "Tests failed with exit code $exit_code"
        fi
        generate_test_report
        return 1
    fi
}

step_cleanup() {
    log "🧹 Step 5: Cleaning up resources..."
    
    if "$SCRIPT_DIR/cluster.sh" down; then
        success "Cleanup completed successfully"
    else
        warning "Cleanup completed with warnings"
    fi
}

generate_test_report() {
    log "📊 Generating test report..."
    
    local report_file="${SCRIPT_DIR}/test-report-$(date +%Y%m%d-%H%M%S).md"
    
    cat > "$report_file" <<EOF
# Semantic Cache E2E Test Report

**Date**: $(date)
**Cluster**: $CLUSTER_NAME
**Workflow Log**: $WORKFLOW_LOG

## Deployment Status

\`\`\`bash
$(kubectl get pods -A 2>/dev/null || echo "Cluster not accessible")
\`\`\`

## Service Status

\`\`\`bash
$(kubectl get svc -A 2>/dev/null || echo "Services not accessible")
\`\`\`

## Recent Events

\`\`\`bash
$(kubectl get events --all-namespaces --sort-by='.lastTimestamp' | tail -10 2>/dev/null || echo "Events not accessible")
\`\`\`

## API Health Check

\`\`\`bash
$(curl -s http://localhost:8080/semantic-cache/health 2>/dev/null || echo "API not accessible")
\`\`\`

## Workflow Log Excerpt

\`\`\`
$(tail -20 "$WORKFLOW_LOG" 2>/dev/null || echo "Log not available")
\`\`\`

---
Generated by workflow.sh at $(date)
EOF

    log "Test report generated: $report_file"
}

show_status() {
    log "📊 Current deployment status..."
    
    echo "=========================================="
    echo "🏗️  CLUSTER STATUS"
    echo "=========================================="
    k3d cluster list | grep "$CLUSTER_NAME" || echo "Cluster not found"
    echo
    
    if kubectl cluster-info >/dev/null 2>&1; then
        echo "=========================================="
        echo "📦 PODS STATUS"
        echo "=========================================="
        kubectl get pods -A
        echo
        
        echo "=========================================="
        echo "🌐 SERVICES STATUS"  
        echo "=========================================="
        kubectl get svc -A
        echo
        
        echo "=========================================="
        echo "🔗 INGRESS STATUS"
        echo "=========================================="
        kubectl get ingress -A
        echo
        
        echo "=========================================="
        echo "❤️  API HEALTH CHECK"
        echo "=========================================="
        if curl -s --max-time 5 "http://localhost:8080/semantic-cache/health"; then
            echo
            echo "✅ API is responding"
        else
            echo "❌ API is not responding"
        fi
        echo
        
        echo "=========================================="
        echo "🌐 WEB INTERFACE"
        echo "=========================================="
        if curl -s --max-time 5 "http://localhost:8080/web/" | head -1; then
            echo
            echo "✅ Web interface is accessible"
        else
            echo "❌ Web interface is not accessible"
        fi
    else
        echo "❌ Cluster is not accessible"
    fi
}

show_logs() {
    log "📜 Showing application logs..."
    
    if kubectl cluster-info >/dev/null 2>&1; then
        echo "=========================================="
        echo "🧠 SEMATIC CACHE LOGS"
        echo "=========================================="
        kubectl logs -n app -l app=sematic-cache --tail=50
        echo
        
        echo "=========================================="
        echo "🌐 WEB SERVICE LOGS"
        echo "=========================================="
        kubectl logs -n app -l app=web --tail=20
        echo
        
        echo "=========================================="
        echo "🐘 POSTGRESQL LOGS"
        echo "=========================================="
        kubectl logs -n infra -l app=postgres --tail=20
        echo
        
        echo "=========================================="
        echo "🔴 REDIS LOGS"
        echo "=========================================="
        kubectl logs -n infra -l app=redis --tail=20
    else
        error "Cluster is not accessible"
        return 1
    fi
}

workflow_full() {
    local skip_cleanup="${SKIP_CLEANUP:-false}"
    
    log "🚀 Starting complete end-to-end workflow..."
    log "Workflow will be logged to: $WORKFLOW_LOG"
    echo
    
    # Initialize log file
    echo "=== Semantic Cache E2E Workflow Log ===" > "$WORKFLOW_LOG"
    echo "Started at: $(date)" >> "$WORKFLOW_LOG"
    echo >> "$WORKFLOW_LOG"
    
    # Execute workflow steps
    step_setup || return 1
    step_build || return 1  
    step_deploy || return 1
    step_test || return 1
    
    success "🎉 Complete workflow executed successfully!"
    
    if [ "$skip_cleanup" = "false" ]; then
        echo
        warning "Cleaning up cluster in 10 seconds... (Ctrl+C to cancel)"
        sleep 10
        step_cleanup
    else
        log "Skipping cleanup (--skip-cleanup specified)"
        log "To cleanup manually: $0 cleanup"
    fi
}

workflow_reset() {
    log "🔄 Starting complete reset workflow..."
    
    step_cleanup || true  # Don't fail if cleanup fails
    step_setup || return 1
    step_build || return 1
    step_deploy || return 1
    
    success "🎉 Reset workflow completed successfully!"
}

main() {
    local commands=()
    local verbose=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -v|--verbose)
                verbose=true
                set -x
                shift
                ;;
            --skip-cleanup)
                export SKIP_CLEANUP=true
                shift
                ;;
            --quick)
                export QUICK_MODE=true
                shift
                ;;
            --timeout)
                export TEST_TIMEOUT="$2"
                shift 2
                ;;
            full|setup|build|deploy|test|quick-test|cleanup|status|logs|reset)
                commands+=("$1")
                shift
                ;;
            *)
                error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Default to full workflow if no commands specified
    if [ ${#commands[@]} -eq 0 ]; then
        commands=("full")
    fi
    
    # Check prerequisites
    if ! check_prerequisites; then
        exit 1
    fi
    
    # Execute commands
    for cmd in "${commands[@]}"; do
        case $cmd in
            full)
                workflow_full
                ;;
            setup)
                step_setup
                ;;
            build)
                step_build
                ;;
            deploy)
                step_deploy
                ;;
            test)
                step_test
                ;;
            quick-test)
                export QUICK_MODE=true
                step_test
                ;;
            cleanup)
                step_cleanup
                ;;
            status)
                show_status
                ;;
            logs)
                show_logs
                ;;
            reset)
                workflow_reset
                ;;
        esac
        
        if [ $? -ne 0 ]; then
            error "Command '$cmd' failed"
            exit 1
        fi
    done
    
    success "All commands completed successfully!"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi