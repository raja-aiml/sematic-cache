#!/usr/bin/env bash
# Common shell functions library for DevOps scripts
# Source this file in your scripts: source "$(dirname "$0")/lib/common.sh"

set -euo pipefail

# === COLOR OUTPUT ===
if [[ -t 1 ]] && [[ "${NO_COLOR:-}" != "true" ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    MAGENTA='\033[0;35m'
    CYAN='\033[0;36m'
    WHITE='\033[1;37m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    MAGENTA=''
    CYAN=''
    WHITE=''
    NC=''
fi

# === LOGGING FUNCTIONS ===
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${CYAN}[DEBUG]${NC} $*" >&2
    fi
}

log_success() {
    echo -e "${GREEN}✅${NC} $*" >&2
}

log_failure() {
    echo -e "${RED}❌${NC} $*" >&2
}

# === ERROR HANDLING ===
die() {
    log_error "$@"
    exit 1
}

trap_error() {
    local line_no=$1
    log_error "Error occurred in script at line: $line_no"
    exit 1
}

# Set error trap
trap 'trap_error ${LINENO}' ERR

# === UTILITY FUNCTIONS ===
# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Verify required tools are installed
verify_tools() {
    local tools=("$@")
    local missing=()
    
    for tool in "${tools[@]}"; do
        if ! command_exists "$tool"; then
            missing+=("$tool")
        fi
    done
    
    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        log_info "Please install the missing tools and try again"
        return 1
    fi
    
    log_debug "All required tools are installed: ${tools[*]}"
    return 0
}

# Get OS type
get_os() {
    case "$(uname -s)" in
        Linux*)  echo "linux";;
        Darwin*) echo "darwin";;
        CYGWIN*|MINGW*|MSYS*) echo "windows";;
        *)       echo "unknown";;
    esac
}

# Get architecture
get_arch() {
    case "$(uname -m)" in
        x86_64|amd64)  echo "amd64";;
        arm64|aarch64) echo "arm64";;
        i386|i686)     echo "386";;
        *)             echo "unknown";;
    esac
}

# === DOCKER FUNCTIONS ===
# Check if Docker is running
is_docker_running() {
    docker info >/dev/null 2>&1
}

# Wait for container to be healthy
wait_for_container() {
    local container=$1
    local timeout=${2:-60}
    
    log_info "Waiting for container $container to be healthy..."
    
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null | grep -q "healthy"; then
            log_success "Container $container is healthy"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    
    log_error "Container $container failed to become healthy within ${timeout}s"
    return 1
}

# === KUBERNETES FUNCTIONS ===
# Check if kubectl context exists
kubectl_context_exists() {
    kubectl config get-contexts -o name | grep -q "^$1$"
}

# Wait for deployment to be ready
wait_for_deployment() {
    local deployment=$1
    local namespace=${2:-default}
    local timeout=${3:-300}
    
    log_info "Waiting for deployment $deployment in namespace $namespace..."
    
    if kubectl rollout status deployment/"$deployment" -n "$namespace" --timeout="${timeout}s"; then
        log_success "Deployment $deployment is ready"
        return 0
    else
        log_error "Deployment $deployment failed to become ready"
        return 1
    fi
}

# Wait for service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local timeout=${3:-60}
    
    log_info "Waiting for $host:$port to be ready..."
    
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            log_success "Service $host:$port is ready"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    
    log_error "Service $host:$port not ready after ${timeout}s"
    return 1
}

# === HTTP FUNCTIONS ===
# Wait for HTTP endpoint to be ready
wait_for_http() {
    local url=$1
    local timeout=${2:-60}
    local expected_code=${3:-200}
    
    log_info "Waiting for HTTP endpoint $url..."
    
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
        if [ "$code" = "$expected_code" ]; then
            log_success "HTTP endpoint $url is ready (${code})"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    
    log_error "HTTP endpoint $url not ready after ${timeout}s"
    return 1
}

# Make HTTP request with retries
http_retry() {
    local url=$1
    local max_attempts=${2:-3}
    local delay=${3:-2}
    
    for attempt in $(seq 1 "$max_attempts"); do
        log_debug "HTTP request attempt $attempt/$max_attempts: $url"
        
        if curl -sS "$url"; then
            return 0
        fi
        
        if [ "$attempt" -lt "$max_attempts" ]; then
            log_warn "Request failed, retrying in ${delay}s..."
            sleep "$delay"
        fi
    done
    
    log_error "HTTP request failed after $max_attempts attempts: $url"
    return 1
}

# === CONFIRMATION PROMPTS ===
# Ask for confirmation
confirm() {
    local message=${1:-"Continue?"}
    local default=${2:-"n"}
    
    local prompt
    if [[ "$default" =~ ^[Yy]$ ]]; then
        prompt="$message [Y/n]: "
    else
        prompt="$message [y/N]: "
    fi
    
    read -r -p "$prompt" response
    response=${response:-$default}
    
    [[ "$response" =~ ^[Yy]$ ]]
}

# === TIMING FUNCTIONS ===
# Measure execution time
measure_time() {
    local start_time
    start_time=$(date +%s)
    
    # Execute the command
    "$@"
    local exit_code=$?
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_info "Execution time: ${duration}s"
    return $exit_code
}

# === CLEANUP FUNCTIONS ===
# Register cleanup function
CLEANUP_FUNCTIONS=()
register_cleanup() {
    CLEANUP_FUNCTIONS+=("$1")
}

# Execute cleanup functions
execute_cleanup() {
    log_info "Executing cleanup functions..."
    
    for func in "${CLEANUP_FUNCTIONS[@]}"; do
        log_debug "Running cleanup: $func"
        $func || log_warn "Cleanup function failed: $func"
    done
}

# Set cleanup trap
trap execute_cleanup EXIT

# === EXPORT FUNCTIONS ===
# Export all functions so they're available to subshells
export -f log_info log_warn log_error log_debug log_success log_failure
export -f die command_exists verify_tools get_os get_arch
export -f is_docker_running wait_for_container
export -f kubectl_context_exists wait_for_deployment wait_for_service
export -f wait_for_http http_retry
export -f confirm measure_time
export -f register_cleanup execute_cleanup

# === MAIN CHECK ===
# Prevent direct execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    log_error "This script should be sourced, not executed directly"
    echo "Usage: source ${BASH_SOURCE[0]}"
    exit 1
fi

log_debug "Common functions library loaded"