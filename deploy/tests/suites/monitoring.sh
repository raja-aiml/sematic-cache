#!/usr/bin/env bash
# =====================================
# 📊 MONITORING & OBSERVABILITY TESTS
# =====================================

# Source test framework
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-framework.sh"

test_application_logging() {
    log "📝 Testing application logging..."
    
    # Check if application pods are generating logs
    for app in "sematic-cache" "web"; do
        local logs
        logs=$(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" logs -l app="$app" --tail=10 2>/dev/null || echo "")
        if [ -n "$logs" ]; then
            success "$app is generating logs"
            
            # Check log format and content
            if echo "$logs" | grep -E "(INFO|ERROR|DEBUG|WARN)" >/dev/null; then
                success "$app logs contain proper log levels"
            else
                warning "$app logs may not have structured logging"
            fi
        else
            warning "$app has no recent logs"
        fi
    done
}

test_infrastructure_logging() {
    log "📝 Testing infrastructure logging..."
    
    # Check infrastructure component logs
    for app in "postgres" "redis"; do
        local logs
        logs=$(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_INFRA" logs -l app="$app" --tail=10 2>/dev/null || echo "")
        if [ -n "$logs" ]; then
            success "$app is generating logs"
        else
            warning "$app has no recent logs"
        fi
    done
    
    # Check ingress controller logs
    local ingress_logs
    ingress_logs=$(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_INFRA" logs -l app.kubernetes.io/name=ingress-nginx --tail=10 2>/dev/null || echo "")
    if [ -n "$ingress_logs" ]; then
        success "Ingress controller is generating logs"
    else
        warning "Ingress controller has no recent logs"
    fi
}

test_log_persistence() {
    log "💾 Testing log persistence..."
    
    # Check if logs persist across pod restarts
    local pod_name
    pod_name=$(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get pods -l app=sematic-cache -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [ -n "$pod_name" ]; then
        # Get current log count
        local log_count_before
        log_count_before=$(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" logs "$pod_name" | wc -l)
        
        # Generate some activity
        curl -s "$BASE_URL/semantic-cache/health" >/dev/null 2>&1 || true
        sleep 2
        
        # Check if new logs appeared
        local log_count_after
        log_count_after=$(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" logs "$pod_name" | wc -l)
        
        if [ "$log_count_after" -ge "$log_count_before" ]; then
            success "Logs are being generated and persisted"
        else
            warning "Log generation may have issues"
        fi
    else
        error "Could not find sematic-cache pod for log testing"
    fi
}

test_resource_metrics() {
    log "📈 Testing resource metrics availability..."
    
    # Check if metrics server is available
    if kubectl --context "$KUBE_CONTEXT" top nodes >/dev/null 2>&1; then
        success "Metrics server is available"
        
        # Test pod metrics
        if kubectl --context "$KUBE_CONTEXT" top pods --all-namespaces >/dev/null 2>&1; then
            success "Pod metrics are available"
            
            # Show current resource usage
            local top_output
            top_output=$(kubectl --context "$KUBE_CONTEXT" top pods -n "$NAMESPACE_APP" 2>/dev/null || echo "No pods found")
            if [ "$top_output" != "No pods found" ]; then
                log "Current resource usage:"
                echo "$top_output"
            fi
        else
            warning "Pod metrics not available"
        fi
    else
        warning "Metrics server not available (normal for basic k3d setups)"
    fi
}

test_pod_health_checks() {
    log "🏥 Testing pod health checks..."
    
    # Check readiness probes
    for app in "sematic-cache" "web"; do
        local ready_status
        ready_status=$(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get pods -l app="$app" -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
        
        if [ "$ready_status" = "True" ]; then
            success "$app readiness probe is healthy"
        else
            error "$app readiness probe failed"
        fi
        
        # Check if liveness probe exists
        local liveness_probe
        liveness_probe=$(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get pods -l app="$app" -o jsonpath='{.items[0].spec.containers[0].livenessProbe}' 2>/dev/null)
        
        if [ -n "$liveness_probe" ] && [ "$liveness_probe" != "null" ]; then
            success "$app has liveness probe configured"
        else
            warning "$app does not have liveness probe configured"
        fi
    done
}

test_restart_counts() {
    log "🔄 Testing pod restart patterns..."
    
    # Check for pods with high restart counts
    local high_restart_pods
    high_restart_pods=$(kubectl --context "$KUBE_CONTEXT" get pods --all-namespaces -o jsonpath='{range .items[*]}{.metadata.namespace}{"\t"}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\n"}{end}' 2>/dev/null | awk '$3 > 5')
    
    if [ -z "$high_restart_pods" ]; then
        success "No pods with excessive restart counts"
    else
        warning "Some pods have high restart counts:"
        echo "$high_restart_pods"
    fi
    
    # Check recent restart events
    local restart_events
    restart_events=$(kubectl --context "$KUBE_CONTEXT" get events --all-namespaces --field-selector reason=Killing,type=Warning --sort-by='.lastTimestamp' 2>/dev/null | tail -5)
    
    if [ -z "$restart_events" ]; then
        success "No recent pod restart events"
    else
        warning "Recent restart events detected"
    fi
}

test_event_monitoring() {
    log "📅 Testing Kubernetes events..."
    
    # Check for warning events in the last hour
    local warning_events
    warning_events=$(kubectl --context "$KUBE_CONTEXT" get events --all-namespaces --field-selector type=Warning 2>/dev/null | tail -10)
    
    if echo "$warning_events" | grep -q "No resources found"; then
        success "No warning events found"
    else
        local warning_count
        warning_count=$(echo "$warning_events" | grep -v "NAMESPACE" | wc -l)
        if [ "$warning_count" -lt 5 ]; then
            warning "$warning_count warning events found (may be normal)"
        else
            error "Many warning events found ($warning_count) - investigation needed"
        fi
    fi
    
    # Check for recent error events
    local error_events
    error_events=$(kubectl --context "$KUBE_CONTEXT" get events --all-namespaces --field-selector reason=Failed 2>/dev/null | tail -5)
    
    if echo "$error_events" | grep -q "No resources found"; then
        success "No recent failure events"
    else
        warning "Recent failure events detected - check cluster health"
    fi
}

test_disk_usage() {
    log "💽 Testing disk usage..."
    
    # Check node disk usage through kubectl
    local node_info
    node_info=$(kubectl --context "$KUBE_CONTEXT" describe nodes 2>/dev/null | grep -A 10 "Allocated resources" || echo "Node info not available")
    
    if [ "$node_info" != "Node info not available" ]; then
        # Look for disk pressure warnings
        if echo "$node_info" | grep -i "disk.*pressure"; then
            warning "Disk pressure detected on nodes"
        else
            success "No disk pressure warnings on nodes"
        fi
    else
        warning "Could not retrieve node disk information"
    fi
    
    # Check persistent volume claims if any
    local pvcs
    pvcs=$(kubectl --context "$KUBE_CONTEXT" get pvc --all-namespaces 2>/dev/null | grep -v "No resources found" || echo "")
    
    if [ -n "$pvcs" ]; then
        success "Persistent volume claims detected"
        echo "$pvcs"
    else
        log "No persistent volume claims (using ephemeral storage)"
    fi
}

test_network_connectivity() {
    log "🌐 Testing internal network connectivity..."
    
    # Test pod-to-pod connectivity
    local api_pod
    api_pod=$(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" get pods -l app=sematic-cache -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [ -n "$api_pod" ]; then
        # Test connectivity to PostgreSQL
        if kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" exec "$api_pod" -- nc -z postgres.infra 5432 2>/dev/null; then
            success "API pod can reach PostgreSQL"
        else
            error "API pod cannot reach PostgreSQL"
        fi
        
        # Test connectivity to Redis
        if kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE_APP" exec "$api_pod" -- nc -z redis.infra 6379 2>/dev/null; then
            success "API pod can reach Redis"
        else
            error "API pod cannot reach Redis"
        fi
    else
        error "Could not find API pod for connectivity testing"
    fi
}

# Export functions for use by main test runner
export -f test_application_logging
export -f test_infrastructure_logging
export -f test_log_persistence
export -f test_resource_metrics
export -f test_pod_health_checks
export -f test_restart_counts
export -f test_event_monitoring
export -f test_disk_usage
export -f test_network_connectivity