#!/bin/bash

# Smoke Test Script for K3D Blueprint
# Quick validation that essential components are working

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results
PASSED=0
FAILED=0
TESTS=()

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    ((PASSED++))
    TESTS+=("✓ $1")
}

log_failure() {
    echo -e "${RED}[✗]${NC} $1"
    ((FAILED++))
    TESTS+=("✗ $1")
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Test cluster connectivity
test_cluster_connectivity() {
    log_info "Testing cluster connectivity..."
    
    if kubectl cluster-info &>/dev/null; then
        log_success "Cluster is accessible"
    else
        log_failure "Cannot connect to cluster"
        return 1
    fi
}

# Test namespace creation
test_namespaces() {
    log_info "Testing namespaces..."
    
    local expected_namespaces=("infra" "app" "monitoring" "istio-system" "logging")
    
    for ns in "${expected_namespaces[@]}"; do
        if kubectl get namespace "$ns" &>/dev/null; then
            log_success "Namespace $ns exists"
        else
            log_failure "Namespace $ns does not exist"
        fi
    done
}

# Test PostgreSQL deployment
test_postgres() {
    log_info "Testing PostgreSQL deployment..."
    
    # Check deployment exists
    if kubectl get deployment postgres -n infra &>/dev/null; then
        log_success "PostgreSQL deployment exists"
    else
        log_failure "PostgreSQL deployment not found"
        return 1
    fi
    
    # Check pod is running
    if kubectl get pods -n infra -l app=postgres --field-selector=status.phase=Running &>/dev/null; then
        log_success "PostgreSQL pod is running"
    else
        log_failure "PostgreSQL pod is not running"
        return 1
    fi
    
    # Check service exists
    if kubectl get service postgres -n infra &>/dev/null; then
        log_success "PostgreSQL service exists"
    else
        log_failure "PostgreSQL service not found"
    fi
    
    # Test database connectivity
    log_info "Testing PostgreSQL connectivity..."
    if kubectl exec -n infra deployment/postgres -- pg_isready -U cache -d cache &>/dev/null; then
        log_success "PostgreSQL is ready and accepting connections"
    else
        log_failure "PostgreSQL is not ready"
    fi
}

# Test Redis deployment
test_redis() {
    log_info "Testing Redis deployment..."
    
    # Check deployment exists
    if kubectl get deployment redis -n infra &>/dev/null; then
        log_success "Redis deployment exists"
    else
        log_failure "Redis deployment not found"
        return 1
    fi
    
    # Check pod is running
    if kubectl get pods -n infra -l app=redis --field-selector=status.phase=Running &>/dev/null; then
        log_success "Redis pod is running"
    else
        log_failure "Redis pod is not running"
        return 1
    fi
    
    # Check service exists
    if kubectl get service redis -n infra &>/dev/null; then
        log_success "Redis service exists"
    else
        log_failure "Redis service not found"
    fi
    
    # Test Redis connectivity
    log_info "Testing Redis connectivity..."
    if kubectl exec -n infra deployment/redis -- redis-cli ping | grep -q "PONG"; then
        log_success "Redis is responding to ping"
    else
        log_failure "Redis is not responding"
    fi
}

# Test persistent volumes
test_persistent_volumes() {
    log_info "Testing persistent volumes..."
    
    # Check PostgreSQL PVC
    if kubectl get pvc postgres-pvc -n infra &>/dev/null; then
        local postgres_status=$(kubectl get pvc postgres-pvc -n infra -o jsonpath='{.status.phase}')
        if [[ "$postgres_status" == "Bound" ]]; then
            log_success "PostgreSQL PVC is bound"
        else
            log_failure "PostgreSQL PVC is not bound (status: $postgres_status)"
        fi
    else
        log_failure "PostgreSQL PVC not found"
    fi
    
    # Check Redis PVC
    if kubectl get pvc redis-pvc -n infra &>/dev/null; then
        local redis_status=$(kubectl get pvc redis-pvc -n infra -o jsonpath='{.status.phase}')
        if [[ "$redis_status" == "Bound" ]]; then
            log_success "Redis PVC is bound"
        else
            log_failure "Redis PVC is not bound (status: $redis_status)"
        fi
    else
        log_failure "Redis PVC not found"
    fi
}

# Test network policies
test_network_policies() {
    log_info "Testing network policies..."
    
    local policies=("default-deny-all" "allow-dns")
    
    for policy in "${policies[@]}"; do
        if kubectl get networkpolicy "$policy" -n infra &>/dev/null; then
            log_success "Network policy $policy exists in infra namespace"
        else
            log_failure "Network policy $policy not found in infra namespace"
        fi
        
        if kubectl get networkpolicy "$policy" -n app &>/dev/null; then
            log_success "Network policy $policy exists in app namespace"
        else
            log_failure "Network policy $policy not found in app namespace"
        fi
    done
}

# Test resource quotas
test_resource_quotas() {
    log_info "Testing resource quotas..."
    
    if kubectl get resourcequota infra-quota -n infra &>/dev/null; then
        log_success "Resource quota exists in infra namespace"
    else
        log_failure "Resource quota not found in infra namespace"
    fi
    
    if kubectl get resourcequota app-quota -n app &>/dev/null; then
        log_success "Resource quota exists in app namespace"
    else
        log_failure "Resource quota not found in app namespace"
    fi
}

# Test metrics endpoints
test_metrics() {
    log_info "Testing metrics endpoints..."
    
    # Check PostgreSQL metrics
    if kubectl get service postgres-metrics -n infra &>/dev/null; then
        log_success "PostgreSQL metrics service exists"
    else
        log_failure "PostgreSQL metrics service not found"
    fi
    
    # Check Redis metrics
    if kubectl get service redis-metrics -n infra &>/dev/null; then
        log_success "Redis metrics service exists"
    else
        log_failure "Redis metrics service not found"
    fi
}

# Test database functionality
test_database_functionality() {
    log_info "Testing database functionality..."
    
    # Test PostgreSQL vector extension
    if kubectl exec -n infra deployment/postgres -- psql -U cache -d cache -c "SELECT 1" &>/dev/null; then
        log_success "PostgreSQL accepts SQL queries"
        
        # Test vector extension
        if kubectl exec -n infra deployment/postgres -- psql -U cache -d cache -c "SELECT extname FROM pg_extension WHERE extname='vector'" | grep -q "vector"; then
            log_success "PostgreSQL vector extension is installed"
        else
            log_failure "PostgreSQL vector extension not found"
        fi
    else
        log_failure "PostgreSQL does not accept SQL queries"
    fi
    
    # Test Redis functionality
    if kubectl exec -n infra deployment/redis -- redis-cli set test-key "test-value" &>/dev/null; then
        if kubectl exec -n infra deployment/redis -- redis-cli get test-key | grep -q "test-value"; then
            log_success "Redis accepts SET/GET operations"
            kubectl exec -n infra deployment/redis -- redis-cli del test-key &>/dev/null
        else
            log_failure "Redis SET/GET operations failed"
        fi
    else
        log_failure "Redis does not accept commands"
    fi
}

# Print summary
print_summary() {
    echo
    echo "========================================"
    echo "        SMOKE TEST SUMMARY"
    echo "========================================"
    echo
    
    for test in "${TESTS[@]}"; do
        echo "$test"
    done
    
    echo
    echo "Total tests: $((PASSED + FAILED))"
    echo -e "${GREEN}Passed: $PASSED${NC}"
    echo -e "${RED}Failed: $FAILED${NC}"
    echo
    
    if [[ $FAILED -eq 0 ]]; then
        echo -e "${GREEN}🎉 All smoke tests passed!${NC}"
        echo -e "${GREEN}The deployment appears to be working correctly.${NC}"
        return 0
    else
        echo -e "${RED}❌ Some smoke tests failed.${NC}"
        echo -e "${RED}Please check the deployment and try again.${NC}"
        return 1
    fi
}

# Main function
main() {
    echo "========================================"
    echo "    K3D BLUEPRINT SMOKE TESTS"
    echo "========================================"
    echo
    
    test_cluster_connectivity
    test_namespaces
    test_postgres
    test_redis
    test_persistent_volumes
    test_network_policies
    test_resource_quotas
    test_metrics
    test_database_functionality
    
    print_summary
}

main "$@"