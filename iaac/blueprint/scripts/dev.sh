#!/bin/bash
# Main development script for K3D blueprint

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source utilities
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/k8s.sh"
source "$SCRIPT_DIR/lib/validation.sh"
source "$SCRIPT_DIR/lib/istio.sh"
source "$SCRIPT_DIR/lib/monitoring.sh"

# Default values
SCENARIO="${SCENARIO:-development}"
CLUSTER_NAME="${CLUSTER_NAME:-k3d-blueprint}"

# Show help
show_help() {
    cat << EOF
K3D Blueprint Development Script

Usage: $0 [command] [options]

Commands:
    setup           Create cluster and deploy base infrastructure
    deploy          Deploy a specific scenario
    validate        Run validation tests
    monitor         Show monitoring endpoints
    clean           Clean up resources
    help            Show this help message

Options:
    --scenario      Scenario to deploy (default: development)
                    Options: minimal, development, service-mesh, monitoring-only, full-stack
    --cluster-name  K3D cluster name (default: k3d-blueprint)

Examples:
    $0 setup
    $0 deploy --scenario full-stack
    $0 validate
    $0 clean

EOF
}

# Setup cluster and base infrastructure
setup() {
    log_info "Setting up K3D blueprint environment"
    
    # Create cluster
    "$SCRIPT_DIR/cluster/cluster.sh" create
    
    # Deploy base infrastructure
    log_info "Deploying base infrastructure"
    apply_kustomization "$ROOT_DIR/infra/base"
    
    # Wait for base services
    wait_for_deployment "postgres" "default" 300
    wait_for_deployment "redis" "default" 300
    
    log_success "Setup completed"
}

# Deploy scenario
deploy() {
    local scenario="${1:-$SCENARIO}"
    local scenario_dir="$ROOT_DIR/scenarios/$scenario"
    
    if [ ! -d "$scenario_dir" ]; then
        log_error "Scenario not found: $scenario"
        exit 1
    fi
    
    log_info "Deploying scenario: $scenario"
    
    # Apply scenario kustomization
    if [ -f "$scenario_dir/kustomization.yaml" ]; then
        apply_kustomization "$scenario_dir"
    else
        log_warning "No kustomization.yaml found in $scenario_dir"
    fi
    
    # Special handling for specific scenarios
    case "$scenario" in
        service-mesh)
            wait_for_istio
            enable_istio_injection "default"
            ;;
        monitoring-only|full-stack)
            wait_for_monitoring
            ;;
    esac
    
    log_success "Scenario $scenario deployed"
}

# Run validation tests
validate() {
    log_info "Running validation tests"
    
    # Basic connectivity tests
    validate_deployment "postgres"
    validate_deployment "redis"
    
    # Check PVCs
    validate_pvc "postgres-data"
    validate_pvc "redis-data"
    
    # Run test suite if available
    if [ -d "$ROOT_DIR/validation-kit/tests" ]; then
        "$ROOT_DIR/validation-kit/scripts/run-tests.sh"
    fi
    
    log_success "Validation completed"
}

# Show monitoring endpoints
monitor() {
    log_info "Monitoring endpoints:"
    
    echo ""
    echo "Grafana: http://localhost:3000 (admin/admin)"
    echo "Prometheus: http://localhost:9090"
    echo "Loki: http://localhost:3100"
    
    if istio_installed; then
        echo "Kiali: http://localhost:20001"
        echo "Jaeger: http://localhost:16686"
    fi
    
    echo ""
    echo "To access services, run:"
    echo "  kubectl port-forward -n monitoring svc/grafana 3000:3000"
    echo "  kubectl port-forward -n monitoring svc/prometheus 9090:9090"
}

# Clean up resources
clean() {
    log_info "Cleaning up K3D blueprint environment"
    
    # Delete cluster
    "$SCRIPT_DIR/cluster/cluster.sh" delete
    
    log_success "Cleanup completed"
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --scenario)
                SCENARIO="$2"
                shift 2
                ;;
            --cluster-name)
                CLUSTER_NAME="$2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done
    
    echo "$@"
}

# Main function
main() {
    local remaining_args=$(parse_args "$@")
    set -- $remaining_args
    
    local command="${1:-help}"
    
    case "$command" in
        setup)
            setup
            ;;
        deploy)
            deploy
            ;;
        validate)
            validate
            ;;
        monitor)
            monitor
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"