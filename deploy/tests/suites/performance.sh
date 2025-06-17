#!/usr/bin/env bash
# =====================================
# ⚡ PERFORMANCE TESTS
# =====================================

# Source test framework
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-framework.sh"

test_response_time() {
    log "⏱️  Testing API response time..."
    
    local total_time=0
    local successful_requests=0
    local max_requests=10
    
    for i in $(seq 1 $max_requests); do
        local start_time
        local end_time
        local response_time
        
        start_time=$(date +%s%N)
        if curl -s --max-time 10 "$BASE_URL/semantic-cache/health" >/dev/null 2>&1; then
            end_time=$(date +%s%N)
            response_time=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
            total_time=$((total_time + response_time))
            ((successful_requests++))
        fi
    done
    
    if [ $successful_requests -gt 0 ]; then
        local avg_response_time=$((total_time / successful_requests))
        if [ $avg_response_time -lt 1000 ]; then
            success "Average response time: ${avg_response_time}ms (excellent)"
        elif [ $avg_response_time -lt 3000 ]; then
            success "Average response time: ${avg_response_time}ms (good)"
        else
            warning "Average response time: ${avg_response_time}ms (may need optimization)"
        fi
    else
        error "No successful requests for response time measurement"
    fi
}

test_concurrent_performance() {
    log "🔄 Testing concurrent request performance..."
    
    local start_time
    local end_time
    local duration
    
    start_time=$(date +%s)
    
    # Run multiple concurrent requests
    local pids=()
    local success_file="/tmp/perf_test_success_$$"
    echo "0" > "$success_file"
    
    for i in {1..20}; do
        {
            local test_data="{\"prompt\":\"Performance test prompt $i\",\"answer\":\"Performance test answer $i\",\"modelName\":\"test\"}"
            if curl -s --max-time 30 -X POST "$BASE_URL/semantic-cache/set" \
                -H "Content-Type: application/json" \
                -d "$test_data" >/dev/null 2>&1; then
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
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    local successful
    successful=$(cat "$success_file")
    rm -f "$success_file"
    
    if [ $duration -lt 30 ] && [ $successful -ge 15 ]; then
        success "Concurrent performance test passed: ${successful}/20 requests in ${duration}s"
    elif [ $successful -ge 10 ]; then
        warning "Concurrent performance: ${successful}/20 requests in ${duration}s (may need optimization)"
    else
        error "Poor concurrent performance: ${successful}/20 requests in ${duration}s"
    fi
}

test_memory_usage() {
    log "💾 Testing memory usage..."
    
    # Check if metrics server is available
    if kubectl --context "$KUBE_CONTEXT" top pods --all-namespaces >/dev/null 2>&1; then
        local memory_usage
        memory_usage=$(kubectl --context "$KUBE_CONTEXT" top pods -n "$NAMESPACE_APP" -l app=sematic-cache --no-headers | awk '{print $3}' | head -1)
        
        if [ -n "$memory_usage" ]; then
            # Extract numeric value (remove 'Mi' suffix)
            local memory_value
            memory_value=$(echo "$memory_usage" | sed 's/Mi$//')
            
            if [ "$memory_value" -lt 512 ]; then
                success "Memory usage within limits: $memory_usage"
            else
                warning "High memory usage: $memory_usage (consider optimization)"
            fi
        else
            warning "Could not retrieve memory usage metrics"
        fi
    else
        warning "Metrics server not available for memory usage testing"
    fi
}

test_cpu_usage() {
    log "🔥 Testing CPU usage..."
    
    # Check if metrics server is available
    if kubectl --context "$KUBE_CONTEXT" top pods --all-namespaces >/dev/null 2>&1; then
        local cpu_usage
        cpu_usage=$(kubectl --context "$KUBE_CONTEXT" top pods -n "$NAMESPACE_APP" -l app=sematic-cache --no-headers | awk '{print $2}' | head -1)
        
        if [ -n "$cpu_usage" ]; then
            # Extract numeric value (remove 'm' suffix)
            local cpu_value
            cpu_value=$(echo "$cpu_usage" | sed 's/m$//')
            
            if [ "$cpu_value" -lt 500 ]; then
                success "CPU usage within limits: $cpu_usage"
            else
                warning "High CPU usage: $cpu_usage (consider optimization)"
            fi
        else
            warning "Could not retrieve CPU usage metrics"
        fi
    else
        warning "Metrics server not available for CPU usage testing"
    fi
}

test_load_handling() {
    log "📈 Testing load handling capabilities..."
    
    # Gradually increase load and measure success rate
    local load_levels=(5 10 20 30)
    
    for load in "${load_levels[@]}"; do
        log "Testing with $load concurrent requests..."
        
        local pids=()
        local success_file="/tmp/load_test_${load}_$$"
        echo "0" > "$success_file"
        
        local start_time
        start_time=$(date +%s)
        
        for i in $(seq 1 $load); do
            {
                if curl -s --max-time 10 "$BASE_URL/semantic-cache/health" >/dev/null 2>&1; then
                    local current
                    current=$(cat "$success_file")
                    echo $((current + 1)) > "$success_file"
                fi
            } &
            pids+=($!)
        done
        
        # Wait for all requests
        for pid in "${pids[@]}"; do
            wait $pid
        done
        
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        local successful
        successful=$(cat "$success_file")
        rm -f "$success_file"
        
        local success_rate=$((successful * 100 / load))
        
        if [ $success_rate -ge 90 ]; then
            success "Load $load: ${success_rate}% success rate in ${duration}s"
        elif [ $success_rate -ge 70 ]; then
            warning "Load $load: ${success_rate}% success rate in ${duration}s"
        else
            error "Load $load: ${success_rate}% success rate in ${duration}s (load handling issues)"
        fi
        
        # Brief pause between load tests
        sleep 2
    done
}

test_cache_performance() {
    log "🗄️  Testing cache performance..."
    
    local cache_key="performance-test-key"
    local test_data="{\"prompt\":\"$cache_key\",\"answer\":\"Performance test answer\",\"modelName\":\"test\"}"
    
    # Set cache entry
    if curl -s -X POST "$BASE_URL/semantic-cache/set" \
        -H "Content-Type: application/json" \
        -d "$test_data" >/dev/null 2>&1; then
        
        # Measure cache retrieval time
        local total_time=0
        local cache_hits=0
        
        for i in {1..10}; do
            local start_time
            local end_time
            
            start_time=$(date +%s%N)
            local get_data="{\"prompt\":\"$cache_key\"}"
            if curl -s -X POST "$BASE_URL/semantic-cache/get" \
                -H "Content-Type: application/json" \
                -d "$get_data" >/dev/null 2>&1; then
                end_time=$(date +%s%N)
                local response_time=$(( (end_time - start_time) / 1000000 ))
                total_time=$((total_time + response_time))
                ((cache_hits++))
            fi
        done
        
        if [ $cache_hits -gt 0 ]; then
            local avg_cache_time=$((total_time / cache_hits))
            if [ $avg_cache_time -lt 500 ]; then
                success "Cache retrieval performance: ${avg_cache_time}ms average (excellent)"
            elif [ $avg_cache_time -lt 1000 ]; then
                success "Cache retrieval performance: ${avg_cache_time}ms average (good)"
            else
                warning "Cache retrieval performance: ${avg_cache_time}ms average (may need optimization)"
            fi
        else
            error "No successful cache retrievals for performance testing"
        fi
    else
        error "Failed to set cache entry for performance testing"
    fi
}

# Export functions for use by main test runner
export -f test_response_time
export -f test_concurrent_performance
export -f test_memory_usage
export -f test_cpu_usage
export -f test_load_handling
export -f test_cache_performance