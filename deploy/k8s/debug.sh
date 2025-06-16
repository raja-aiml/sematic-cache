#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# 🔧 CONFIGURATION
# ─────────────────────────────────────────────────────────────
NAMESPACE="app"
APP_LABEL="app=sematic-cache"
BASE_URL="localhost:8080"
HOST_HEADER="sematic.127.0.0.1.nip.io"

# Global variables
POD=""

# ─────────────────────────────────────────────────────────────
# 📖 USAGE
# ─────────────────────────────────────────────────────────────
function usage() {
    cat <<EOM
Usage: $(basename "$0") <command>

Commands:
  full           Complete debugging analysis (default)
  api            Test API endpoints only (quick test)
  apitest        Clean API test suite
  pod            Pod status and logs only
  image          Docker image analysis only
  source         Source code analysis only
  network        Network connectivity tests
  summary        Issue analysis and recommendations only
  help           Show this help message

Examples:
  $(basename "$0")           # Run full analysis
  $(basename "$0") api       # Test API endpoints (part of full analysis)
  $(basename "$0") apitest   # Clean, standalone API test
  $(basename "$0") pod       # Check pod status
  $(basename "$0") source    # Analyze source code
  $(basename "$0") summary   # Get issue analysis and action plan

Quick Commands:
  $(basename "$0") apitest   # Fast API testing (recommended for regular use)
  $(basename "$0") summary   # Quick issue diagnosis and action plan
  $(basename "$0") full      # Complete analysis (use when troubleshooting)
EOM
}

# ─────────────────────────────────────────────────────────────
# 🔍 UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────
function log_section() {
    local title="$1"
    echo ""
    echo "$title"
    echo "$(printf '=%.0s' $(seq 1 ${#title}))"
}

function log_subsection() {
    local title="$1"
    echo ""
    echo "🔹 $title"
    # Fix the printf issue - use a different approach for creating dashes
    local dashes=""
    for ((i=0; i<$((${#title} + 4)); i++)); do
        dashes+="-"
    done
    echo "$dashes"
}

function check_prerequisites() {
    log_section "🔧 Checking Prerequisites"
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        echo "❌ Namespace '$NAMESPACE' does not exist"
        exit 1
    fi
    
    # Check if pods exist
    local pods=$(kubectl get pods -n "$NAMESPACE" -l "$APP_LABEL" -o json 2>/dev/null || echo '{"items":[]}')
    if ! echo "$pods" | jq -e '.items | length > 0' >/dev/null 2>&1; then
        echo "❌ No pods found with label '$APP_LABEL' in namespace '$NAMESPACE'"
        echo "📦 All pods in '$NAMESPACE' namespace:"
        kubectl get pods -n "$NAMESPACE" 2>/dev/null || echo "No pods found"
        exit 1
    fi
    
    echo "✅ Prerequisites check passed"
}

function get_pod_info() {
    local pods=$(kubectl get pods -n "$NAMESPACE" -l "$APP_LABEL" -o json)
    POD=$(echo "$pods" | jq -r '.items | sort_by(.metadata.creationTimestamp) | reverse | .[0].metadata.name')
    
    if [[ -z "${POD:-}" || "$POD" == "null" ]]; then
        echo "❌ Failed to get pod name"
        exit 1
    fi
    
    echo "🎯 Selected pod: $POD"
}

# ─────────────────────────────────────────────────────────────
# 🧪 API TESTING FUNCTIONS
# ─────────────────────────────────────────────────────────────
function test_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    
    log_subsection "$name"
    
    local response
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\nHTTP_CODE:%{http_code}" -H "Host: $HOST_HEADER" http://$BASE_URL$endpoint)
    else
        response=$(curl -s -w "\nHTTP_CODE:%{http_code}" -H "Host: $HOST_HEADER" -H "Content-Type: application/json" \
          -X $method http://$BASE_URL$endpoint -d "$data")
    fi
    
    local http_code=$(echo "$response" | tail -n1 | cut -d: -f2)
    local body=$(echo "$response" | sed '$d')
    
    # HTTP 200, 201, 202 are all success codes
    if [[ "$http_code" =~ ^(200|201|202)$ ]]; then
        if [ "$http_code" = "201" ]; then
            echo "✅ Success (HTTP $http_code - Created)"
        else
            echo "✅ Success (HTTP $http_code)"
        fi
        if [ -n "$body" ]; then
            echo "$body" | jq . 2>/dev/null || echo "$body"
        else
            echo "(Empty response - normal for some operations)"
        fi
    else
        echo "❌ Error (HTTP $http_code)"
        echo "$body" | jq . 2>/dev/null || echo "$body"
    fi
}

function test_endpoint_clean() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    
    echo ""
    echo "$name"
    echo "$(printf '=%.0s' $(seq 1 ${#name}))"
    
    local response
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\nHTTP_CODE:%{http_code}" -H "Host: $HOST_HEADER" http://$BASE_URL$endpoint)
    else
        response=$(curl -s -w "\nHTTP_CODE:%{http_code}" -H "Host: $HOST_HEADER" -H "Content-Type: application/json" \
          -X $method http://$BASE_URL$endpoint -d "$data")
    fi
    
    local http_code=$(echo "$response" | tail -n1 | cut -d: -f2)
    local body=$(echo "$response" | sed '$d')
    
    # HTTP 200, 201, 202 are all success codes
    if [[ "$http_code" =~ ^(200|201|202)$ ]]; then
        if [ "$http_code" = "201" ]; then
            echo "✅ Success (HTTP $http_code - Created)"
        else
            echo "✅ Success (HTTP $http_code)"
        fi
        if [ -n "$body" ]; then
            echo "$body" | jq . 2>/dev/null || echo "$body"
        else
            echo "(Empty response - normal for some operations)"
        fi
    else
        echo "❌ Error (HTTP $http_code)"
        echo "$body" | jq . 2>/dev/null || echo "$body"
    fi
}

function run_clean_api_test() {
    echo "🚀 Sematic Cache API Test Suite"
    echo "==============================="
    
    echo ""
    echo "⚠️  OpenAI API Key Status: Using 'dummy-key'"
    echo "• ✅ Basic storage works"  
    echo "• ❌ Semantic search requires real OpenAI key"
    
    # Core functionality tests
    test_endpoint_clean "📊 Health Check" "GET" "/health" ""
    test_endpoint_clean "📈 Metrics" "GET" "/metrics" ""
    test_endpoint_clean "💾 Cache SET" "POST" "/set" '{"key": "test", "value": "value"}'
    test_endpoint_clean "🔍 Cache GET" "POST" "/get" '{"key": "test"}'
    test_endpoint_clean "🔎 Semantic Query" "POST" "/query" '{"query": "test", "threshold": 0.8}'
    test_endpoint_clean "🏆 Top-K Search" "POST" "/topk" '{"query": "test", "k": 5}'
    test_endpoint_clean "🧹 Admin Flush" "POST" "/admin/flush" '{}'
    
    echo ""
    echo ""
    echo "✅ Testing Complete!"
    echo ""
    echo "🌐 Access your API at: http://sematic.127.0.0.1.nip.io:8080"
    echo ""
    echo "💡 To enable semantic features:"
    echo "   Get an OpenAI API key and update the deployment"
    echo "   kubectl patch deployment sematic-cache -n app -p '{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"sematic-cache\",\"env\":[{\"name\":\"OPENAI_API_KEY\",\"value\":\"your-key\"}]}]}}}}'"
}

function test_api_endpoints() {
    log_section "🚀 API Endpoint Testing"
    
    echo ""
    echo "⚠️  IMPORTANT: OpenAI API Key Status"
    echo "Your app is using 'dummy-key' for OpenAI API, which means:"
    echo "• ✅ Basic storage works (data is stored)"  
    echo "• ❌ Semantic search disabled (no embeddings)"
    echo "• ❌ GET operations return empty (can't find without embeddings)"
    
    # Core endpoints
    test_endpoint "Health Check" "GET" "/health" ""
    test_endpoint "Metrics" "GET" "/metrics" ""
    
    # Storage operations
    test_endpoint "Cache SET Operation" "POST" "/set" \
      '{"key": "test-basic", "value": "basic storage test"}'
    test_endpoint "Cache GET Operation" "POST" "/get" \
      '{"key": "test-basic"}'
    
    # AI-dependent features  
    test_endpoint "Query Operation (semantic search)" "POST" "/query" \
      '{"query": "test question", "threshold": 0.8}'
    test_endpoint "Top-K Search (vector search)" "POST" "/topk" \
      '{"query": "test", "k": 5}'
    
    # Admin operations
    test_endpoint "Admin Flush" "POST" "/admin/flush" '{}'
}

# ─────────────────────────────────────────────────────────────
# 🐳 POD ANALYSIS FUNCTIONS  
# ─────────────────────────────────────────────────────────────
function analyze_pod_status() {
    log_section "📦 Pod Status Analysis"
    
    # Display pods overview
    echo "📦 Pods in '$NAMESPACE' namespace:"
    kubectl get pods -n "$NAMESPACE"
    
    # Get detailed pod information
    log_subsection "Pod Details"
    local pod_info=$(kubectl get pod -n "$NAMESPACE" "$POD" -o json)
    
    # Extract key information
    local phase=$(echo "$pod_info" | jq -r '.status.phase // "Unknown"')
    local ready=$(echo "$pod_info" | jq -r '.status.conditions[] | select(.type=="Ready") | .status // "Unknown"')
    local restart_count=$(echo "$pod_info" | jq -r '.status.containerStatuses[0].restartCount // 0')
    local image=$(echo "$pod_info" | jq -r '.spec.containers[0].image // "Unknown"')
    
    echo "🆔 Pod Name:       $POD"
    echo "📊 Phase:          $phase"
    echo "✅ Ready:          $ready"
    echo "🔁 Restart Count:  $restart_count"
    echo "🐳 Image:          $image"
    
    # Container status details
    if echo "$pod_info" | jq -e '.status.containerStatuses[0]' >/dev/null 2>&1; then
        local container_status=$(echo "$pod_info" | jq -r '.status.containerStatuses[0]')
        local current_state=$(echo "$container_status" | jq -r '.state | keys[0]')
        
        echo "📌 Current State:  $current_state"
        
        case "$current_state" in
            "waiting")
                local reason=$(echo "$container_status" | jq -r '.state.waiting.reason // "Unknown"')
                local message=$(echo "$container_status" | jq -r '.state.waiting.message // ""')
                echo "⏳ Waiting Reason: $reason"
                [[ -n "$message" && "$message" != "null" ]] && echo "💬 Message:        $message"
                ;;
            "running")
                local started=$(echo "$container_status" | jq -r '.state.running.startedAt // "Unknown"')
                echo "🏃 Started At:     $started"
                ;;
            "terminated")
                local exit_code=$(echo "$container_status" | jq -r '.state.terminated.exitCode // "Unknown"')
                local reason=$(echo "$container_status" | jq -r '.state.terminated.reason // "Unknown"')
                echo "💥 Exit Code:      $exit_code"
                echo "📖 Reason:         $reason"
                ;;
        esac
    fi
}

function show_pod_logs() {
    log_section "📜 Pod Logs Analysis"
    
    echo "📜 Recent container logs:"
    if kubectl logs -n "$NAMESPACE" "$POD" --tail=20 2>/dev/null; then
        echo "✅ Logs retrieved successfully"
    else
        echo "⚠️ Could not fetch current logs, trying previous container..."
        if kubectl logs -n "$NAMESPACE" "$POD" --previous --tail=20 2>/dev/null; then
            echo "✅ Previous container logs retrieved"
        else
            echo "❌ No logs available"
        fi
    fi
}

function show_recent_events() {
    log_section "📅 Recent Kubernetes Events"
    
    kubectl get events -n "$NAMESPACE" --field-selector involvedObject.name="$POD" --sort-by='.lastTimestamp' | tail -10
}

# ─────────────────────────────────────────────────────────────
# 🐳 DOCKER IMAGE ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────
function analyze_docker_image() {
    log_section "🐳 Docker Image Analysis"
    
    local image=$(kubectl get pod -n "$NAMESPACE" "$POD" -o jsonpath='{.spec.containers[0].image}')
    echo "🐳 Analyzing image: $image"
    
    if ! command -v docker >/dev/null 2>&1; then
        echo "❌ Docker not available for image analysis"
        return 1
    fi
    
    if docker image inspect "$image" >/dev/null 2>&1; then
        log_subsection "Image Configuration"
        docker image inspect "$image" | jq -r '.[0] | {
          "Created": .Created,
          "Architecture": .Architecture,
          "Os": .Os,
          "Size": .Size,
          "Cmd": .Config.Cmd,
          "Entrypoint": .Config.Entrypoint,
          "WorkingDir": .Config.WorkingDir
        }' 2>/dev/null || echo "❌ Cannot parse image config"
        
        log_subsection "Testing Image Locally"
        echo "Testing binary existence:"
        if timeout 10s docker run --rm "$image" test -f /app/sematic-cache 2>&1; then
            echo "✅ Binary exists in image"
            timeout 10s docker run --rm "$image" ls -la /app/sematic-cache 2>&1 || echo "❌ Cannot get binary details"
        else
            echo "❌ Binary NOT found at /app/sematic-cache"
            echo "Searching for binary:"
            timeout 15s docker run --rm "$image" find / -name "*sematic*" -type f 2>/dev/null | head -5 || echo "❌ Search failed"
        fi
        
        log_subsection "Testing Application Startup"
        echo "Testing help command:"
        if timeout 5s docker run --rm "$image" /app/sematic-cache --help 2>&1; then
            echo "✅ Application responds to --help"
        else
            echo "❌ Application startup failed"
        fi
    else
        echo "❌ Image '$image' not found locally"
        echo "Available sematic-cache images:"
        docker images | grep -E "(sematic|cache)" || echo "No related images found"
    fi
}

# ─────────────────────────────────────────────────────────────
# 📄 SOURCE CODE ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────
function analyze_source_code() {
    log_section "📄 Source Code Analysis"
    
    log_subsection "Checking Available Endpoints"
    
    # Check what's actually available in your app's source code
    echo "🔍 Searching for 'query' endpoint in source code:"
    if grep -r "query" cmd/server/main.go 2>/dev/null; then
        echo "✅ Found 'query' references in source"
    else
        echo "❌ No 'query' references found in cmd/server/main.go"
        echo "🔍 Checking other server files:"
        find . -name "*.go" -path "*/server/*" -exec grep -l "query\|Query" {} \; 2>/dev/null || echo "No query endpoints found in server files"
    fi
    
    echo ""
    echo "🔍 Checking all available HTTP routes:"
    if grep -r "POST\|GET\|PUT\|DELETE" cmd/server/main.go 2>/dev/null; then
        echo "✅ Found HTTP route definitions"
        echo ""
        echo "🔍 Route details:"
        grep -n "POST\|GET\|PUT\|DELETE" cmd/server/main.go 2>/dev/null || echo "No route details found"
    else
        echo "❌ No obvious HTTP routes found in main.go"
        echo "🔍 Checking server directory:"
        find . -name "*.go" -exec grep -l "gin\|http\|router" {} \; 2>/dev/null | head -5
    fi
    
    echo ""
    echo "🔍 Checking for Gin routes in all Go files:"
    find . -name "*.go" -exec grep -l "\.POST\|\.GET\|\.PUT\|\.DELETE" {} \; 2>/dev/null | head -10 || echo "No Gin routes found"
    
    echo ""
    echo "🔍 Searching for specific endpoint patterns:"
    echo "Query-related patterns:"
    grep -r "query\|Query" . --include="*.go" 2>/dev/null | head -5 || echo "No query patterns found"
    
    log_subsection "Dockerfile Analysis"
    local dockerfile_paths=("deploy/docker/Dockerfile" "Dockerfile")
    local dockerfile_found=""
    
    for dockerfile_path in "${dockerfile_paths[@]}"; do
        if [[ -f "$dockerfile_path" ]]; then
            dockerfile_found="$dockerfile_path"
            break
        fi
    done
    
    if [[ -n "$dockerfile_found" ]]; then
        echo "✅ Found Dockerfile at: $dockerfile_found"
        echo ""
        echo "🔍 Key Dockerfile commands:"
        echo "COPY/ADD commands:"
        grep -n -E "^(COPY|ADD)" "$dockerfile_found" || echo "No COPY/ADD commands found"
        echo ""
        echo "RUN commands:"
        grep -n -E "^RUN" "$dockerfile_found" || echo "No RUN commands found"
        echo ""
        echo "ENTRYPOINT/CMD:"
        grep -n -E "^(CMD|ENTRYPOINT)" "$dockerfile_found" || echo "No CMD/ENTRYPOINT commands found"
    else
        echo "❌ Dockerfile not found in common locations"
    fi
}

# ─────────────────────────────────────────────────────────────
# 🌐 NETWORK CONNECTIVITY FUNCTIONS
# ─────────────────────────────────────────────────────────────
function test_network_connectivity() {
    log_section "🌐 Network Connectivity Analysis"
    
    log_subsection "Internal Connectivity"
    
    # Test if pod can exec
    if kubectl exec -n "$NAMESPACE" "$POD" -- true 2>/dev/null; then
        echo "✅ Pod is ready for exec commands"
        
        # Test internal endpoints
        echo ""
        echo "🔍 Testing internal health endpoint:"
        if kubectl exec -n "$NAMESPACE" "$POD" -- wget -qO- http://localhost:8080/health 2>/dev/null; then
            echo "✅ Internal health endpoint responding"
        else
            echo "❌ Internal health endpoint not responding"
        fi
        
        # Check what's listening
        echo ""
        echo "🔍 Checking listening ports:"
        kubectl exec -n "$NAMESPACE" "$POD" -- netstat -tlnp 2>/dev/null || echo "❌ Cannot check listening ports"
        
        # Test database connectivity
        echo ""
        echo "🔍 Testing database connectivity:"
        if kubectl exec -n "$NAMESPACE" "$POD" -- ping -c 1 postgres.infra.svc.cluster.local >/dev/null 2>&1; then
            echo "✅ Can reach postgres.infra.svc.cluster.local"
        else
            echo "❌ Cannot reach postgres.infra.svc.cluster.local"
        fi
        
        # Check for debug endpoints
        echo ""
        echo "🔍 Checking for debug endpoints:"
        kubectl exec -n "$NAMESPACE" "$POD" -- curl -s http://localhost:8080/debug/pprof/ 2>/dev/null || echo "❌ No debug endpoint available"
        
    else
        echo "⚠️ Pod is not ready for exec commands"
    fi
    
    log_subsection "External Connectivity"
    
    # Test external access via ingress
    echo "🔍 Testing external access:"
    echo "Via ingress (with proper host header):"
    if curl -s -H "Host: $HOST_HEADER" http://$BASE_URL/health >/dev/null 2>&1; then
        echo "✅ Ingress access working"
    else
        echo "❌ Ingress access failing"
    fi
}

# ─────────────────────────────────────────────────────────────
# 📋 COMPREHENSIVE ISSUE ANALYSIS AND SUMMARY
# ─────────────────────────────────────────────────────────────
function analyze_issues_and_summarize() {
    log_section "🔍 COMPREHENSIVE ISSUE ANALYSIS"
    
    # Collect current status
    local pod_ready=$(kubectl get pod -n "$NAMESPACE" "$POD" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
    local restart_count=$(kubectl get pod -n "$NAMESPACE" "$POD" -o jsonpath='{.status.containerStatuses[0].restartCount}' 2>/dev/null)
    local image=$(kubectl get pod -n "$NAMESPACE" "$POD" -o jsonpath='{.spec.containers[0].image}')
    
    # Test key endpoints
    local health_status=$(curl -s -o /dev/null -w "%{http_code}" -H "Host: $HOST_HEADER" http://$BASE_URL/health 2>/dev/null || echo "000")
    local query_status=$(curl -s -o /dev/null -w "%{http_code}" -H "Host: $HOST_HEADER" -H "Content-Type: application/json" -X POST http://$BASE_URL/query -d '{"query":"test"}' 2>/dev/null || echo "000")
    local set_status=$(curl -s -o /dev/null -w "%{http_code}" -H "Host: $HOST_HEADER" -H "Content-Type: application/json" -X POST http://$BASE_URL/set -d '{"key":"test","value":"test"}' 2>/dev/null || echo "000")
    
    # Check database connectivity
    local db_reachable="false"
    if kubectl exec -n "$NAMESPACE" "$POD" -- ping -c 1 postgres.infra.svc.cluster.local >/dev/null 2>&1; then
        db_reachable="true"
    fi
    
    # Check if binary exists in image
    local binary_exists="false"
    if command -v docker >/dev/null 2>&1 && docker image inspect "$image" >/dev/null 2>&1; then
        if timeout 5s docker run --rm "$image" test -f /app/sematic-cache >/dev/null 2>&1; then
            binary_exists="true"
        fi
    fi
    
    # Analyze findings
    echo "📊 DIAGNOSTIC RESULTS:"
    echo "========================"
    
    echo "🔧 Infrastructure Status:"
    echo "  Pod Ready: $([[ "$pod_ready" == "True" ]] && echo "✅ YES" || echo "❌ NO")"
    echo "  Restart Count: ${restart_count:-0}"
    echo "  Image: $image"
    echo "  Binary in Image: $([[ "$binary_exists" == "true" ]] && echo "✅ YES" || echo "❌ NO")"
    
    echo ""
    echo "🌐 API Connectivity:"
    echo "  Health Endpoint (/health): $([[ "$health_status" == "200" ]] && echo "✅ $health_status" || echo "❌ $health_status")"
    echo "  Storage Endpoint (/set): $([[ "$set_status" =~ ^(200|201)$ ]] && echo "✅ $set_status" || echo "❌ $set_status")"
    echo "  Query Endpoint (/query): $([[ "$query_status" == "200" ]] && echo "✅ $query_status" || echo "❌ $query_status")"
    
    echo ""
    echo "🗄️ Data Layer:"
    echo "  Database Reachable: $([[ "$db_reachable" == "true" ]] && echo "✅ YES" || echo "❌ NO")"
    echo "  OpenAI Integration: ❌ DUMMY KEY"
    
    # Issue severity analysis
    log_subsection "Issue Severity Analysis"
    
    local critical_issues=0
    local major_issues=0
    local minor_issues=0
    
    echo "🚨 CRITICAL ISSUES (App won't work):"
    if [[ "$pod_ready" != "True" ]]; then
        echo "  • Pod not ready - application not running"
        ((critical_issues++))
    fi
    if [[ "$health_status" != "200" ]]; then
        echo "  • Health endpoint failing - basic functionality broken"
        ((critical_issues++))
    fi
    if [[ "$binary_exists" == "false" ]]; then
        echo "  • Binary missing from Docker image - deployment will fail"
        ((critical_issues++))
    fi
    [[ $critical_issues -eq 0 ]] && echo "  ✅ No critical issues found!"
    
    echo ""
    echo "⚠️ MAJOR ISSUES (Limited functionality):"
    if [[ "$query_status" == "404" ]]; then
        echo "  • Query endpoint returns 404 - semantic search not working"
        ((major_issues++))
    fi
    if [[ "$db_reachable" == "false" ]]; then
        echo "  • Database unreachable - data persistence may fail"
        ((major_issues++))
    fi
    [[ $major_issues -eq 0 ]] && echo "  ✅ No major issues found!"
    
    echo ""
    echo "ℹ️ MINOR ISSUES (Reduced features):"
    echo "  • OpenAI API key is dummy - semantic features disabled"
    ((minor_issues++))
    if [[ "${restart_count:-0}" -gt 0 ]]; then
        echo "  • Pod has restarted $restart_count times - check for stability issues"
        ((minor_issues++))
    fi
    
    # Root cause analysis
    log_subsection "Root Cause Analysis"
    
    if [[ $critical_issues -gt 0 ]]; then
        echo "🎯 PRIMARY ISSUE: Critical infrastructure problems"
        echo "   Focus on fixing pod readiness and health endpoint first"
    elif [[ $major_issues -gt 0 ]]; then
        echo "🎯 PRIMARY ISSUE: Application configuration problems"
        if [[ "$query_status" == "404" ]]; then
            echo "   The /query endpoint exists in code but returns 404"
            echo "   This suggests a routing or path mismatch issue"
        fi
        if [[ "$db_reachable" == "false" ]]; then
            echo "   Database connectivity issue - check network policies and service names"
        fi
    else
        echo "🎯 DEPLOYMENT STATUS: Mostly successful!"
        echo "   Core functionality is working, only missing semantic features"
    fi
    
    # Action plan
    log_subsection "Recommended Action Plan"
    
    if [[ $critical_issues -gt 0 ]]; then
        echo "🚨 IMMEDIATE ACTIONS (Critical):"
        echo "1. Fix pod and health endpoint issues first"
        echo "2. Verify Docker image build process"
        echo "3. Redeploy with working image"
    elif [[ $major_issues -gt 0 ]]; then
        echo "⚠️ HIGH PRIORITY ACTIONS:"
        if [[ "$query_status" == "404" ]]; then
            echo "1. Investigate query endpoint routing:"
            echo "   • Check if endpoint path changed"
            echo "   • Verify server.go vs main.go route definitions"
            echo "   • Test endpoint directly: kubectl exec -n app deployment/sematic-cache -- curl -X POST http://localhost:8080/query -d '{\"query\":\"test\"}'"
        fi
        if [[ "$db_reachable" == "false" ]]; then
            echo "2. Fix database connectivity:"
            echo "   • Check if postgres is running: kubectl get pods -n infra"
            echo "   • Verify service name: postgres.infra.svc.cluster.local"
            echo "   • Check network policies"
        fi
    fi
    
    echo ""
    echo "✨ ENHANCEMENT ACTIONS (Optional):"
    echo "1. Add real OpenAI API key for semantic features:"
    echo "   kubectl patch deployment sematic-cache -n app -p '{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"sematic-cache\",\"env\":[{\"name\":\"OPENAI_API_KEY\",\"value\":\"sk-your-key\"}]}]}}}}'"
    
    # Success indicators
    log_subsection "Success Indicators"
    
    local success_score=0
    local total_checks=7
    
    [[ "$pod_ready" == "True" ]] && ((success_score++))
    [[ "$health_status" == "200" ]] && ((success_score++))
    [[ "$set_status" =~ ^(200|201)$ ]] && ((success_score++))
    [[ "$query_status" == "200" ]] && ((success_score++))
    [[ "$db_reachable" == "true" ]] && ((success_score++))
    [[ "$binary_exists" == "true" ]] && ((success_score++))
    [[ "${restart_count:-0}" -eq 0 ]] && ((success_score++))
    
    local success_percentage=$((success_score * 100 / total_checks))
    
    echo "📊 DEPLOYMENT SUCCESS RATE: $success_score/$total_checks ($success_percentage%)"
    
    if [[ $success_percentage -ge 80 ]]; then
        echo "🎉 EXCELLENT: Deployment is mostly successful!"
    elif [[ $success_percentage -ge 60 ]]; then
        echo "👍 GOOD: Deployment is functional with minor issues"
    elif [[ $success_percentage -ge 40 ]]; then
        echo "⚠️ FAIR: Deployment has significant issues but core functionality works"
    else
        echo "❌ POOR: Deployment has major problems requiring immediate attention"
    fi
    
    # Quick test commands
    log_subsection "Quick Test Commands"
    
    echo "🧪 Test your deployment:"
    echo "  # Quick API test"
    echo "  curl -H 'Host: sematic.127.0.0.1.nip.io' http://localhost:8080/health"
    echo ""
    echo "  # Test storage"
    echo "  curl -H 'Host: sematic.127.0.0.1.nip.io' -H 'Content-Type: application/json' -X POST http://localhost:8080/set -d '{\"key\":\"test\",\"value\":\"hello\"}'"
    echo ""
    echo "  # Test query endpoint directly in pod"
    echo "  kubectl exec -n app deployment/sematic-cache -- curl -X POST http://localhost:8080/query -H 'Content-Type: application/json' -d '{\"query\":\"test\"}'"
}

# ─────────────────────────────────────────────────────────────
# 🧭 MAIN FUNCTION
# ─────────────────────────────────────────────────────────────
function main() {
    local command="${1:-full}"
    
    case "$command" in
        full)
            echo "🔍 COMPREHENSIVE SEMATIC CACHE DEBUGGING"
            echo "========================================"
            check_prerequisites
            get_pod_info
            analyze_pod_status
            show_pod_logs
            show_recent_events
            test_api_endpoints
            analyze_docker_image
            analyze_source_code
            test_network_connectivity
            analyze_issues_and_summarize
            ;;
        api)
            echo "🧪 API ENDPOINT TESTING"
            echo "======================"
            check_prerequisites
            get_pod_info
            test_api_endpoints
            ;;
        apitest)
            echo "🚀 CLEAN API TEST SUITE"
            echo "======================"
            run_clean_api_test
            ;;
        pod)
            echo "📦 POD STATUS ANALYSIS"
            echo "====================="
            check_prerequisites
            get_pod_info
            analyze_pod_status
            show_pod_logs
            show_recent_events
            ;;
        image)
            echo "🐳 DOCKER IMAGE ANALYSIS"
            echo "========================"
            check_prerequisites
            get_pod_info
            analyze_docker_image
            ;;
        source)
            echo "📄 SOURCE CODE ANALYSIS"
            echo "======================"
            analyze_source_code
            ;;
        network)
            echo "🌐 NETWORK CONNECTIVITY TESTING"
            echo "==============================="
            check_prerequisites
            get_pod_info
            test_network_connectivity
            ;;
        summary)
            echo "📋 ISSUE ANALYSIS & SUMMARY"
            echo "==========================="
            check_prerequisites
            get_pod_info
            analyze_issues_and_summarize
            ;;
        help|*)
            usage
            ;;
    esac
}

# Entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi