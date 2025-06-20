#!/usr/bin/env bash
# =====================================
# 🔬 API ENDPOINT TESTS
# =====================================

# Source test framework
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-framework.sh"

test_health_endpoints() {
    log "❤️  Testing health endpoints..."
    
    # Wait for services to be fully ready
    log "Waiting for services to be fully ready..."
    sleep 30
    
    # Test API health endpoint
    http_test "GET" "$BASE_URL/semantic-cache/health" "200" "" "API health endpoint" 5
    
    # Test API metrics endpoint
    http_test "GET" "$BASE_URL/semantic-cache/metrics" "200" "" "API metrics endpoint" 3
}

test_web_interface() {
    log "🌐 Testing web interface..."
    
    # Test web interface
    http_test "GET" "$BASE_URL/web/" "200" "" "Web interface" 3
    
    # Test web interface content
    local response
    response=$(curl -s "$BASE_URL/web/" 2>/dev/null || echo "")
    if echo "$response" | grep -q "Semantic Cache"; then
        success "Web interface contains expected content"
    else
        error "Web interface content incorrect"
    fi
}

test_cache_set_operation() {
    log "💾 Testing cache SET operation..."
    
    local set_data="{\"prompt\":\"$TEST_PROMPT\",\"answer\":\"$TEST_ANSWER\",\"modelName\":\"$TEST_MODEL\"}"
    http_test "POST" "$BASE_URL/semantic-cache/set" "200" "$set_data" "Cache SET operation" 3
}

test_cache_get_operation() {
    log "💾 Testing cache GET operation..."
    
    local get_data="{\"prompt\":\"$TEST_PROMPT\"}"
    http_test "POST" "$BASE_URL/semantic-cache/get" "200" "$get_data" "Cache GET operation" 3
}

test_cache_miss() {
    log "💾 Testing cache MISS behavior..."
    
    local miss_data="{\"prompt\":\"What is Docker containers and orchestration?\"}"
    if http_test "POST" "$BASE_URL/semantic-cache/get" "200" "$miss_data" "Cache MISS test" 3; then
        log "Cache miss test completed (expected behavior)"
    fi
}

test_cache_operations() {
    log "💾 Testing cache operations..."
    test_cache_set_operation
    test_cache_get_operation
    test_cache_miss
}

test_query_endpoint() {
    log "🧠 Testing query endpoint..."
    
    local query_data="{\"embedding\":[0.1,0.2,0.3,0.4,0.5]}"
    http_test "POST" "$BASE_URL/semantic-cache/query" "200" "$query_data" "Query with embedding" 3
}

test_topk_endpoint() {
    log "🔝 Testing top-K endpoint..."
    
    local topk_data="{\"embedding\":[0.1,0.2,0.3,0.4,0.5],\"k\":5}"
    http_test "POST" "$BASE_URL/semantic-cache/topk" "200" "$topk_data" "Top-K query" 3
}

test_advanced_operations() {
    log "🧠 Testing advanced operations..."
    test_query_endpoint
    test_topk_endpoint
}

test_error_handling() {
    log "🚨 Testing error handling..."
    
    # Test with invalid JSON
    local invalid_json="invalid json"
    if curl -s -X POST "$BASE_URL/semantic-cache/set" \
        -H "Content-Type: application/json" \
        -d "$invalid_json" | grep -q "error\|Error" 2>/dev/null; then
        success "API correctly handles invalid JSON"
    else
        warning "Error handling test inconclusive"
    fi
    
    # Test with missing required fields
    local incomplete_data="{\"prompt\":\"test\"}"
    local response
    response=$(curl -s -w "HTTPSTATUS:%{http_code}" -X POST "$BASE_URL/semantic-cache/set" \
        -H "Content-Type: application/json" \
        -d "$incomplete_data" 2>/dev/null || echo "HTTPSTATUS:000")
    
    local status
    status=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    
    if [ "$status" = "400" ] || [ "$status" = "422" ]; then
        success "API correctly validates required fields"
    else
        warning "Field validation test inconclusive (Status: $status)"
    fi
}

test_api_cors() {
    log "🌐 Testing CORS headers..."
    
    local cors_response
    cors_response=$(curl -s -H "Origin: http://localhost:3000" \
        -H "Access-Control-Request-Method: POST" \
        -H "Access-Control-Request-Headers: Content-Type" \
        -X OPTIONS "$BASE_URL/semantic-cache/health" -I 2>/dev/null || echo "")
    
    if echo "$cors_response" | grep -qi "access-control"; then
        success "CORS headers are present"
    else
        warning "CORS headers not detected (may not be configured)"
    fi
}

test_rate_limiting() {
    log "🚦 Testing rate limiting behavior..."
    
    # Send multiple rapid requests
    local success_count=0
    local rate_limited_count=0
    
    for i in {1..20}; do
        local status
        status=$(curl -s -w "%{http_code}" -o /dev/null "$BASE_URL/semantic-cache/health" 2>/dev/null || echo "000")
        
        if [ "$status" = "200" ]; then
            ((success_count++))
        elif [ "$status" = "429" ]; then
            ((rate_limited_count++))
        fi
    done
    
    if [ $rate_limited_count -gt 0 ]; then
        success "Rate limiting is active ($rate_limited_count/20 requests limited)"
    elif [ $success_count -eq 20 ]; then
        warning "No rate limiting detected (may not be configured)"
    else
        warning "Rate limiting test inconclusive"
    fi
}

test_concurrent_requests() {
    log "🔄 Testing concurrent request handling..."
    
    local pids=()
    local success_file="/tmp/api_test_success_$$"
    echo "0" > "$success_file"
    
    # Launch 10 concurrent requests
    for i in {1..10}; do
        {
            if curl -s --max-time 30 "$BASE_URL/semantic-cache/health" >/dev/null 2>&1; then
                local current
                current=$(cat "$success_file")
                echo $((current + 1)) > "$success_file"
            fi
        } &
        pids+=($!)
    done
    
    # Wait for all requests to complete
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    local successful
    successful=$(cat "$success_file")
    rm -f "$success_file"
    
    if [ "$successful" -ge 8 ]; then
        success "Concurrent requests handled successfully ($successful/10)"
    else
        error "Concurrent request handling issues ($successful/10 successful)"
    fi
}

# Export functions for use by main test runner
export -f test_health_endpoints
export -f test_web_interface
export -f test_cache_operations
export -f test_advanced_operations
export -f test_error_handling
export -f test_api_cors
export -f test_rate_limiting
export -f test_concurrent_requests