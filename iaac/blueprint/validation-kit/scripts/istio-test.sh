#!/bin/bash
# Istio service mesh validation tests

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source utilities
source "$ROOT_DIR/scripts/lib/common.sh"
source "$ROOT_DIR/scripts/lib/k8s.sh"
source "$ROOT_DIR/scripts/lib/istio.sh"

# Test Istio installation
test_istio() {
    log_info "Starting Istio validation tests"
    
    # Check if Istio is installed
    if ! istio_installed; then
        log_error "Istio not installed"
        exit 1
    fi
    
    # Wait for Istio components
    wait_for_istio
    
    # Test Istio control plane
    log_info "Testing Istio control plane..."
    if kubectl exec -n istio-system deployment/istiod -- pilot-discovery version; then
        log_success "Istio control plane is running"
    else
        log_error "Istio control plane check failed"
        exit 1
    fi
    
    # Check sidecar injection
    log_info "Checking sidecar injection..."
    local namespaces_with_injection=$(kubectl get namespaces -l istio-injection=enabled --no-headers | wc -l)
    log_info "Found $namespaces_with_injection namespaces with injection enabled"
    
    # Validate Istio configuration
    validate_istio_config "default"
    
    # Check mTLS status
    check_mtls_status "default"
    
    # Deploy test application with sidecar
    log_info "Deploying test application with Istio sidecar..."
    kubectl apply -f "$ROOT_DIR/validation-kit/client-connections/istio-proxy-test.yaml"
    
    # Wait for test pod
    if wait_for_pod "app=istio-proxy-test" "default" 300; then
        log_success "Istio test application deployed"
        
        # Check if sidecar is injected
        local containers=$(kubectl get pod -l app=istio-proxy-test -o jsonpath='{.items[0].spec.containers[*].name}')
        if echo "$containers" | grep -q "istio-proxy"; then
            log_success "Istio sidecar injected successfully"
        else
            log_error "Istio sidecar not found"
            kubectl delete -f "$ROOT_DIR/validation-kit/client-connections/istio-proxy-test.yaml"
            exit 1
        fi
        
        # Test sidecar functionality
        log_info "Testing sidecar proxy..."
        kubectl exec -l app=istio-proxy-test -c istio-proxy -- \
            curl -s localhost:15000/clusters | head -20
        
        # Clean up test application
        kubectl delete -f "$ROOT_DIR/validation-kit/client-connections/istio-proxy-test.yaml"
    else
        log_error "Failed to deploy Istio test application"
        exit 1
    fi
    
    # Run Istio-specific tests
    log_info "Running Istio test jobs..."
    local istio_test="$ROOT_DIR/validation-kit/tests/integration/istio-test.yaml"
    if [ -f "$istio_test" ]; then
        kubectl apply -f "$istio_test"
        
        if kubectl wait --for=condition=complete job/istio-integration-test --timeout=300s; then
            log_success "Istio integration test completed"
            kubectl logs job/istio-integration-test
            kubectl delete job istio-integration-test
        else
            log_error "Istio integration test failed"
            kubectl logs job/istio-integration-test || true
            kubectl delete job istio-integration-test || true
            exit 1
        fi
    fi
    
    # Test traffic management
    log_info "Testing Istio traffic management..."
    kubectl apply -f "$ROOT_DIR/validation-kit/seed-data/istio/traffic-scenarios.yaml"
    kubectl apply -f "$ROOT_DIR/validation-kit/seed-data/istio/policy-tests.yaml"
    
    # Clean up test resources
    kubectl delete -f "$ROOT_DIR/validation-kit/seed-data/istio/traffic-scenarios.yaml" || true
    kubectl delete -f "$ROOT_DIR/validation-kit/seed-data/istio/policy-tests.yaml" || true
    
    log_success "All Istio tests passed"
}

# Main
main() {
    test_istio
}

main "$@"