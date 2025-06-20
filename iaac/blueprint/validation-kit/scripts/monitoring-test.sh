#!/bin/bash
# Monitoring validation tests

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source utilities
source "$ROOT_DIR/scripts/lib/common.sh"
source "$ROOT_DIR/scripts/lib/k8s.sh"
source "$ROOT_DIR/scripts/lib/monitoring.sh"

# Test monitoring stack
test_monitoring() {
    log_info "Starting monitoring validation tests"
    
    # Check if monitoring is installed
    if ! monitoring_installed; then
        log_error "Monitoring stack not installed"
        exit 1
    fi
    
    # Wait for monitoring components
    wait_for_monitoring
    
    # Test Prometheus
    log_info "Testing Prometheus..."
    if kubectl exec -n monitoring deployment/prometheus -- wget -q -O- http://localhost:9090/-/healthy | grep -q "Prometheus is Healthy"; then
        log_success "Prometheus is healthy"
    else
        log_error "Prometheus health check failed"
        exit 1
    fi
    
    # Test Grafana
    log_info "Testing Grafana..."
    if kubectl exec -n monitoring deployment/grafana -- curl -s http://localhost:3000/api/health | grep -q "ok"; then
        log_success "Grafana is healthy"
    else
        log_error "Grafana health check failed"
        exit 1
    fi
    
    # Test Loki
    log_info "Testing Loki..."
    if kubectl exec -n monitoring deployment/loki -- wget -q -O- http://localhost:3100/ready | grep -q "ready"; then
        log_success "Loki is ready"
    else
        log_warning "Loki readiness check failed (may not be deployed)"
    fi
    
    # Check metrics collection
    log_info "Checking metrics collection..."
    local targets=$(kubectl exec -n monitoring deployment/prometheus -- \
        curl -s http://localhost:9090/api/v1/targets | \
        jq '.data.activeTargets | length')
    
    if [ "$targets" -gt 0 ]; then
        log_success "Found $targets active Prometheus targets"
    else
        log_error "No active Prometheus targets found"
        exit 1
    fi
    
    # Run observability tests
    log_info "Running observability test jobs..."
    for test_file in "$ROOT_DIR/validation-kit/tests/observability"/*.yaml; do
        if [ -f "$test_file" ]; then
            local test_name=$(basename "$test_file" .yaml)
            log_info "Running test: $test_name"
            
            # Apply test job
            kubectl apply -f "$test_file"
            
            # Wait for job completion
            if kubectl wait --for=condition=complete job/"$test_name" --timeout=300s; then
                log_success "Test $test_name completed"
                
                # Get job logs
                kubectl logs job/"$test_name"
                
                # Clean up
                kubectl delete job "$test_name"
            else
                log_error "Test $test_name failed"
                kubectl logs job/"$test_name" || true
                kubectl delete job "$test_name" || true
                exit 1
            fi
        fi
    done
    
    log_success "All monitoring tests passed"
}

# Main
main() {
    test_monitoring
}

main "$@"