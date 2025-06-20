#!/bin/bash
# Generate test report

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source utilities
source "$ROOT_DIR/scripts/lib/common.sh"
source "$ROOT_DIR/scripts/lib/k8s.sh"

# Report file
REPORT_FILE="${REPORT_FILE:-$ROOT_DIR/validation-report.md}"

# Generate header
generate_header() {
    cat > "$REPORT_FILE" << EOF
# K3D Blueprint Validation Report

Generated on: $(date)

## Summary

EOF
}

# Check component status
check_components() {
    log_info "Checking component status..."
    
    echo "### Component Status" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "| Component | Namespace | Status | Ready |" >> "$REPORT_FILE"
    echo "|-----------|-----------|--------|-------|" >> "$REPORT_FILE"
    
    # Core components
    local components=(
        "postgres:default"
        "redis:default"
        "prometheus:monitoring"
        "grafana:monitoring"
        "loki:monitoring"
        "istiod:istio-system"
        "istio-ingressgateway:istio-system"
    )
    
    for comp in "${components[@]}"; do
        IFS=':' read -r name namespace <<< "$comp"
        
        if resource_exists "deployment" "$name" "$namespace"; then
            local ready=$(kubectl get deployment "$name" -n "$namespace" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
            local desired=$(kubectl get deployment "$name" -n "$namespace" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
            
            if [ "$ready" = "$desired" ] && [ "$ready" != "0" ]; then
                echo "| $name | $namespace | ✅ Running | $ready/$desired |" >> "$REPORT_FILE"
            else
                echo "| $name | $namespace | ❌ Not Ready | $ready/$desired |" >> "$REPORT_FILE"
            fi
        else
            echo "| $name | $namespace | ❌ Not Found | - |" >> "$REPORT_FILE"
        fi
    done
    
    echo "" >> "$REPORT_FILE"
}

# Check PVC status
check_storage() {
    log_info "Checking storage status..."
    
    echo "### Storage Status" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "| PVC Name | Namespace | Status | Size |" >> "$REPORT_FILE"
    echo "|----------|-----------|--------|------|" >> "$REPORT_FILE"
    
    kubectl get pvc --all-namespaces -o json | jq -r '.items[] | 
        "| \(.metadata.name) | \(.metadata.namespace) | \(.status.phase) | \(.status.capacity.storage // "N/A") |"' >> "$REPORT_FILE"
    
    echo "" >> "$REPORT_FILE"
}

# Check service endpoints
check_services() {
    log_info "Checking service endpoints..."
    
    echo "### Service Endpoints" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "| Service | Namespace | Type | Endpoints |" >> "$REPORT_FILE"
    echo "|---------|-----------|------|-----------|" >> "$REPORT_FILE"
    
    kubectl get services --all-namespaces -o json | jq -r '.items[] | 
        select(.metadata.name != "kubernetes") |
        "| \(.metadata.name) | \(.metadata.namespace) | \(.spec.type) | \(.spec.clusterIP):\(.spec.ports[0].port // "N/A") |"' >> "$REPORT_FILE"
    
    echo "" >> "$REPORT_FILE"
}

# Run test summary
run_test_summary() {
    log_info "Running test summary..."
    
    echo "### Test Results" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    local test_categories=(
        "integration"
        "performance"
        "security"
        "observability"
    )
    
    echo "| Test Category | Tests | Status |" >> "$REPORT_FILE"
    echo "|---------------|-------|--------|" >> "$REPORT_FILE"
    
    for category in "${test_categories[@]}"; do
        local test_count=$(ls "$ROOT_DIR/validation-kit/tests/$category"/*.yaml 2>/dev/null | wc -l)
        if [ "$test_count" -gt 0 ]; then
            echo "| $category | $test_count | ✅ Available |" >> "$REPORT_FILE"
        else
            echo "| $category | 0 | ❌ No tests |" >> "$REPORT_FILE"
        fi
    done
    
    echo "" >> "$REPORT_FILE"
}

# Check resource usage
check_resources() {
    log_info "Checking resource usage..."
    
    echo "### Resource Usage" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    kubectl top nodes 2>/dev/null || echo "Metrics server not available" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "#### Top Pods by CPU" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    kubectl top pods --all-namespaces --sort-by=cpu 2>/dev/null | head -10 || echo "Metrics not available" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "#### Top Pods by Memory" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    kubectl top pods --all-namespaces --sort-by=memory 2>/dev/null | head -10 || echo "Metrics not available" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
}

# Generate recommendations
generate_recommendations() {
    log_info "Generating recommendations..."
    
    echo "## Recommendations" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # Check for missing components
    if ! resource_exists "deployment" "prometheus" "monitoring"; then
        echo "- 🔧 **Monitoring**: Prometheus is not deployed. Consider deploying the monitoring stack for better observability." >> "$REPORT_FILE"
    fi
    
    if ! resource_exists "deployment" "istiod" "istio-system"; then
        echo "- 🔧 **Service Mesh**: Istio is not deployed. Consider enabling service mesh for advanced traffic management and security." >> "$REPORT_FILE"
    fi
    
    # Check PVC status
    local unbound_pvcs=$(kubectl get pvc --all-namespaces -o json | jq -r '.items[] | select(.status.phase != "Bound") | .metadata.name' | wc -l)
    if [ "$unbound_pvcs" -gt 0 ]; then
        echo "- ⚠️  **Storage**: Found $unbound_pvcs unbound PVCs. Check storage provisioner configuration." >> "$REPORT_FILE"
    fi
    
    echo "" >> "$REPORT_FILE"
}

# Main function
main() {
    log_info "Generating validation report..."
    
    # Generate report sections
    generate_header
    check_components
    check_storage
    check_services
    run_test_summary
    check_resources
    generate_recommendations
    
    # Add footer
    echo "---" >> "$REPORT_FILE"
    echo "Report generated by K3D Blueprint validation suite" >> "$REPORT_FILE"
    echo "Location: $REPORT_FILE" >> "$REPORT_FILE"
    
    log_success "Report generated: $REPORT_FILE"
    
    # Display report
    cat "$REPORT_FILE"
}

main "$@"