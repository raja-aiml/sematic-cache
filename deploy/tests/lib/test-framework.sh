#!/usr/bin/env bash
# =====================================
# 🧪 TEST FRAMEWORK
# =====================================

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils.sh"

# HTTP test with retries
http_test() {
    local method="$1"
    local url="$2"
    local expected_status="$3"
    local data="${4:-}"
    local description="$5"
    local retries="${6:-3}"
    
    log "Testing: $description"
    
    for i in $(seq 1 $retries); do
        local response
        local status
        
        if [ -n "$data" ]; then
            response=$(curl -s -w "HTTPSTATUS:%{http_code}" -X "$method" \
                -H "Content-Type: application/json" \
                -d "$data" "$url" 2>/dev/null || echo "HTTPSTATUS:000")
        else
            response=$(curl -s -w "HTTPSTATUS:%{http_code}" -X "$method" "$url" 2>/dev/null || echo "HTTPSTATUS:000")
        fi
        
        status=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
        body=$(echo "$response" | sed 's/HTTPSTATUS:[0-9]*$//')
        
        if [ "$status" = "$expected_status" ]; then
            success "$description (Status: $status)"
            if [ -n "$body" ] && [ "$body" != "null" ]; then
                echo "   Response: $(echo "$body" | head -c 100)..."
            fi
            return 0
        fi
        
        if [ $i -lt $retries ]; then
            warning "Attempt $i failed (Status: $status), retrying..."
            sleep 2
        fi
    done
    
    error "$description (Expected: $expected_status, Got: $status)"
    if [ -n "$body" ]; then
        echo "   Response: $body"
    fi
    return 1
}

# Run test suite
run_test_suite() {
    local suite_name="$1"
    shift
    local tests=("$@")
    
    log "🎯 Running $suite_name test suite..."
    echo "=================================="
    
    for test_func in "${tests[@]}"; do
        if command -v "$test_func" >/dev/null 2>&1; then
            $test_func
        else
            error "Test function '$test_func' not found"
        fi
        echo
    done
}

# Show test summary
show_summary() {
    echo
    echo "=================================="
    log "📊 TEST SUMMARY"
    echo "=================================="
    echo -e "${GREEN}✅ Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}❌ Failed: $TESTS_FAILED${NC}"
    echo -e "${BLUE}📊 Total:  $((TESTS_PASSED + TESTS_FAILED))${NC}"
    
    if [ $TESTS_FAILED -gt 0 ]; then
        echo
        echo -e "${RED}❌ Failed tests:${NC}"
        for test in "${FAILED_TESTS[@]}"; do
            echo -e "${RED}   - $test${NC}"
        done
        echo
        echo -e "${RED}🚨 Some tests failed. Check the output above for details.${NC}"
        return 1
    else
        echo
        echo -e "${GREEN}🎉 All tests passed! Deployment is ready for production.${NC}"
        return 0
    fi
}

# Test if deployment is ready
test_deployment_ready() {
    local namespace="$1"
    local app="$2"
    local timeout="${3:-120}"
    
    if wait_for "[ \"\$(get_pod_phase '$namespace' 'app=$app')\" = 'Running' ]" $timeout; then
        success "$app pod is running"
        
        if wait_for "is_pod_ready '$namespace' 'app=$app'" 60; then
            success "$app pod is ready"
            return 0
        else
            error "$app pod not ready"
            return 1
        fi
    else
        error "$app pod not running"
        return 1
    fi
}

# Test service endpoints
test_service_endpoints() {
    local namespace="$1"
    local service="$2"
    
    if kubectl --context "$KUBE_CONTEXT" -n "$namespace" get svc "$service" >/dev/null 2>&1; then
        success "$service service exists"
        
        local endpoints
        endpoints=$(kubectl --context "$KUBE_CONTEXT" -n "$namespace" get endpoints "$service" -o jsonpath='{.subsets[0].addresses}')
        if [ -n "$endpoints" ] && [ "$endpoints" != "null" ]; then
            success "$service service has endpoints"
            return 0
        else
            error "$service service has no endpoints"
            return 1
        fi
    else
        error "$service service not found"
        return 1
    fi
}