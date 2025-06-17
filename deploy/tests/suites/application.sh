#!/usr/bin/env bash
# =====================================
# 🚀 APPLICATION TESTS
# =====================================

# Source test framework
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-framework.sh"

test_sematic_cache_deployment() {
    log "🧠 Testing Sematic Cache API deployment..."
    test_deployment_ready "$NAMESPACE_APP" "sematic-cache" 120
}

test_web_deployment() {
    log "🌐 Testing Web service deployment..."
    test_deployment_ready "$NAMESPACE_APP" "web" 60
}

test_application_deployments() {
    log "🚀 Testing application deployments..."
    test_sematic_cache_deployment
    test_web_deployment
}

test_sematic_cache_service() {
    log "🔗 Testing Sematic Cache service..."
    test_service_endpoints "$NAMESPACE_APP" "sematic-cache"
}

test_web_service() {
    log "🌐 Testing Web service..."
    test_service_endpoints "$NAMESPACE_APP" "web"
}

test_services() {
    log "🌐 Testing Kubernetes services..."
    test_sematic_cache_service
    test_web_service
}

test_ingress() {
    log "🔗 Testing ingress configuration..."
    
    # Check ingress exists
    if kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get ingress app-ingress >/dev/null 2>&1; then
        success "App ingress exists"
        
        # Check ingress has address (optional, takes time)
        wait_for "kubectl --context '$KUBE_CONTEXT' -n '$NAMESPACE_APP' get ingress app-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}' | grep -v '^$'" 60 5 || true
        
        # Test external connectivity
        if curl -s --max-time 10 "$BASE_URL" >/dev/null 2>&1; then
            success "Ingress is externally accessible"
        else
            warning "Ingress may not be externally accessible yet (normal for new deployments)"
        fi
    else
        error "App ingress not found"
    fi
}

test_application_scaling() {
    log "📈 Testing application scaling..."
    
    # Test scaling sematic-cache deployment
    local original_replicas
    original_replicas=$(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get deployment sematic-cache -o jsonpath='{.spec.replicas}')
    
    # Scale to 2 replicas
    kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" scale deployment sematic-cache --replicas=2 >/dev/null 2>&1
    if wait_for "kubectl --context '$KUBE_CONTEXT' -n '$NAMESPACE_APP' get deployment sematic-cache -o jsonpath='{.status.readyReplicas}' | grep -q '2'" 120; then
        success "Successfully scaled to 2 replicas"
        
        # Scale back to original
        kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" scale deployment sematic-cache --replicas="$original_replicas" >/dev/null 2>&1
        wait_for "kubectl --context '$KUBE_CONTEXT' -n '$NAMESPACE_APP' get deployment sematic-cache -o jsonpath='{.status.readyReplicas}' | grep -q '$original_replicas'" 120
        success "Successfully scaled back to $original_replicas replica(s)"
    else
        error "Failed to scale deployment"
        # Attempt to restore original state
        kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" scale deployment sematic-cache --replicas="$original_replicas" >/dev/null 2>&1
    fi
}

test_config_maps() {
    log "🗺️  Testing ConfigMaps..."
    
    # Check web content ConfigMap
    local web_configmaps
    web_configmaps=$(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get configmap -l app=web --no-headers | wc -l)
    if [ "$web_configmaps" -gt 0 ]; then
        success "Web ConfigMap(s) exist"
    else
        error "Web ConfigMap not found"
    fi
}

test_secrets() {
    log "🔐 Testing Secrets..."
    
    # Check sematic-cache secrets
    if kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get secret sematic-cache-secrets >/dev/null 2>&1; then
        success "Sematic cache secrets exist"
        
        # Verify secret has required keys
        local keys
        keys=$(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get secret sematic-cache-secrets -o jsonpath='{.data}' | grep -o '"[^"]*"' | tr -d '"')
        if echo "$keys" | grep -q "DATABASE_URL" && echo "$keys" | grep -q "OPENAI_API_KEY"; then
            success "Secrets contain required keys"
        else
            error "Secrets missing required keys"
        fi
    else
        error "Sematic cache secrets not found"
    fi
}

# Export functions for use by main test runner
export -f test_application_deployments
export -f test_services
export -f test_ingress
export -f test_application_scaling
export -f test_config_maps
export -f test_secrets
export -f test_sematic_cache_deployment
export -f test_web_deployment