#!/usr/bin/env bash
# =====================================
# 🏗️  INFRASTRUCTURE TESTS
# =====================================

# Source test framework
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-framework.sh"

test_cluster_creation() {
    log "🏗️  Testing cluster creation..."
    
    # Check if cluster exists
    if k3d cluster list | grep -q "$CLUSTER_NAME"; then
        success "Cluster '$CLUSTER_NAME' exists"
    else
        error "Cluster '$CLUSTER_NAME' not found"
        return 1
    fi
    
    # Check cluster health
    if kubectl --context "$KUBE_CONTEXT" cluster-info >/dev/null 2>&1; then
        success "Cluster is accessible"
    else
        error "Cannot connect to cluster"
        return 1
    fi
    
    # Check nodes
    local nodes
    nodes=$(kubectl --context "$KUBE_CONTEXT" get nodes --no-headers | wc -l)
    if [ "$nodes" -gt 0 ]; then
        success "Cluster has $nodes node(s)"
    else
        error "No nodes found in cluster"
        return 1
    fi
}

test_namespaces() {
    log "📁 Testing namespaces..."
    
    for ns in "$NAMESPACE_INFRA" "$NAMESPACE_APP"; do
        if kubectl --context "$KUBE_CONTEXT" get namespace "$ns" >/dev/null 2>&1; then
            success "Namespace '$ns' exists"
        else
            error "Namespace '$ns' not found"
        fi
    done
}

test_postgres() {
    log "🐘 Testing PostgreSQL..."
    
    if test_deployment_ready "$NAMESPACE_INFRA" "postgres" 60; then
        # Test PostgreSQL connectivity
        if kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_INFRA" exec -i deployment/postgres -- psql -U cache -d cache -c "SELECT 1;" >/dev/null 2>&1; then
            success "PostgreSQL is accepting connections"
        else
            error "PostgreSQL connection failed"
        fi
        
        # Test pgvector extension
        if kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_INFRA" exec -i deployment/postgres -- psql -U cache -d cache -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null 2>&1; then
            success "PostgreSQL pgvector extension available"
        else
            error "PostgreSQL pgvector extension failed"
        fi
    fi
}

test_redis() {
    log "🔴 Testing Redis..."
    
    if test_deployment_ready "$NAMESPACE_INFRA" "redis" 60; then
        # Test Redis connectivity
        if kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_INFRA" exec -i deployment/redis -- redis-cli ping | grep -q PONG; then
            success "Redis is responding to ping"
        else
            error "Redis ping failed"
        fi
    fi
}

test_ingress_controller() {
    log "🌐 Testing Ingress Controller..."
    
    if wait_for "kubectl --context '$KUBE_CONTEXT' -n '$NAMESPACE_INFRA' get pod -l app.kubernetes.io/name=ingress-nginx -o jsonpath='{.items[0].status.phase}' | grep -q Running" 120; then
        success "Ingress controller pod is running"
        
        # Test ingress controller service
        if kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_INFRA" get svc ingress-nginx-controller >/dev/null 2>&1; then
            success "Ingress controller service exists"
        else
            error "Ingress controller service not found"
        fi
    else
        error "Ingress controller pod not running"
    fi
}

test_infrastructure_components() {
    log "🔧 Testing all infrastructure components..."
    test_postgres
    test_redis
    test_ingress_controller
}

# Export functions for use by main test runner
export -f test_cluster_creation
export -f test_namespaces
export -f test_infrastructure_components
export -f test_postgres
export -f test_redis
export -f test_ingress_controller