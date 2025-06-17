#!/usr/bin/env bash
# =====================================
# 🔧 UTILITY FUNCTIONS
# =====================================

# Source configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((TESTS_PASSED++))
}

error() {
    echo -e "${RED}❌ $1${NC}"
    ((TESTS_FAILED++))
    FAILED_TESTS+=("$1")
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Wait for condition with timeout
wait_for() {
    local condition="$1"
    local timeout="${2:-60}"
    local interval="${3:-5}"
    local count=0
    
    log "Waiting for: $condition"
    while ! eval "$condition" && [ $count -lt $((timeout / interval)) ]; do
        sleep $interval
        ((count++))
        echo -n "."
    done
    echo
    
    if [ $count -ge $((timeout / interval)) ]; then
        error "Timeout waiting for: $condition"
        return 1
    fi
    success "Condition met: $condition"
    return 0
}

# Check if command exists
check_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        error "Required command '$cmd' not found"
        return 1
    fi
    return 0
}

# Check prerequisites
check_prerequisites() {
    log "🔍 Checking prerequisites..."
    local missing=0
    
    for cmd in kubectl k3d curl; do
        if ! check_command "$cmd"; then
            ((missing++))
        fi
    done
    
    if [ $missing -gt 0 ]; then
        error "$missing required commands are missing"
        return 1
    fi
    
    success "All prerequisites are available"
    return 0
}

# Get pod phase
get_pod_phase() {
    local namespace="$1"
    local selector="$2"
    kubectl --context "$KUBE_CONTEXT" -n "$namespace" get pod -l "$selector" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Unknown"
}

# Get pod ready status
is_pod_ready() {
    local namespace="$1"
    local selector="$2"
    local ready_status
    ready_status=$(kubectl --context "$KUBE_CONTEXT" -n "$namespace" get pod -l "$selector" -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
    [ "$ready_status" = "True" ]
}