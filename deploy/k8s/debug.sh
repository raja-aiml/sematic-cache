#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 SEMATIC CACHE MANAGEMENT TOOL
# ═══════════════════════════════════════════════════════════════════════════════
# Complete tool for managing Kubernetes secrets, deployment, and debugging
# of the Sematic Cache application.
#
# Features:
# - Secrets management (OpenAI API keys, database URLs)
# - Comprehensive debugging and analysis
# - API endpoint testing
# - Pod status monitoring
# - Network connectivity testing
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 🔧 GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
readonly NAMESPACE="app"
readonly SECRET_NAME="sematic-cache-secrets"
readonly CONFIGMAP_NAME="sematic-cache-config"
readonly APP_LABEL="app=sematic-cache"
readonly BASE_URL="localhost:8080"
readonly HOST_HEADER="sematic.127.0.0.1.nip.io"

# Global variables for pod management
POD=""

# ─────────────────────────────────────────────────────────────────────────────
# 📖 USAGE AND HELP
# ─────────────────────────────────────────────────────────────────────────────
function show_main_usage() {
    cat <<EOM
🚀 SEMATIC CACHE MANAGEMENT TOOL
================================

Usage: $(basename "$0") <category> <command> [options]

CATEGORIES:
━━━━━━━━━━
  secrets    Manage Kubernetes secrets and configuration
  debug      Debug and analyze running deployment
  test       Test API endpoints and functionality
  help       Show detailed help for any category

QUICK COMMANDS:
━━━━━━━━━━━━━━━
  $(basename "$0") secrets create     # Create secrets from environment
  $(basename "$0") test api           # Quick API test
  $(basename "$0") debug full         # Complete diagnostic analysis
  $(basename "$0") debug summary      # Quick issue analysis

EXAMPLES:
━━━━━━━━━
  # Initial setup
  export OPENAI_API_KEY="sk-your-key-here"
  $(basename "$0") secrets create

  # Test your deployment
  $(basename "$0") test api

  # Debug issues
  $(basename "$0") debug summary

  # Get detailed help
  $(basename "$0") help secrets
  $(basename "$0") help debug
  $(basename "$0") help test

For detailed usage of any category, use:
  $(basename "$0") help <category>
EOM
}

function show_secrets_help() {
    cat <<EOM
🔐 SECRETS MANAGEMENT COMMANDS
==============================

Usage: $(basename "$0") secrets <command>

COMMANDS:
━━━━━━━━━
  create           Create secrets from environment variables
  update-openai    Update OpenAI API key only
  update-db        Update database URL only
  view             View current secrets (safely decoded)
  delete           Delete all secrets and config
  create-config    Create non-sensitive configuration
  load-env         Load environment variables from .env file
  create-env       Create sample .env file

ENVIRONMENT VARIABLES:
━━━━━━━━━━━━━━━━━━━━━━
  OPENAI_API_KEY   Your OpenAI API key (required for semantic features)
  DATABASE_URL     Database connection string (optional, uses default)

SETUP METHODS:
━━━━━━━━━━━━━━
  Method 1: Direct environment variable
    export OPENAI_API_KEY="sk-your-real-key-here"
    $(basename "$0") secrets create

  Method 2: Using .env file
    $(basename "$0") secrets create-env
    # Edit .env file with your keys
    $(basename "$0") secrets load-env
    $(basename "$0") secrets create

  Method 3: One-liner
    OPENAI_API_KEY="sk-your-key" $(basename "$0") secrets create

EXAMPLES:
━━━━━━━━━
  $(basename "$0") secrets create        # Create secrets
  $(basename "$0") secrets view          # View current secrets
  $(basename "$0") secrets update-openai # Update just the API key
EOM
}

function show_debug_help() {
    cat <<EOM
🔍 DEBUG AND ANALYSIS COMMANDS
===============================

Usage: $(basename "$0") debug <command>

COMMANDS:
━━━━━━━━━
  full             Complete diagnostic analysis (recommended for troubleshooting)
  summary          Quick issue analysis and action plan (recommended for regular use)
  pod              Pod status and logs analysis
  image            Docker image analysis
  source           Source code analysis
  network          Network connectivity testing

COMMAND DESCRIPTIONS:
━━━━━━━━━━━━━━━━━━━━━
  full      - Comprehensive analysis of all components
            - Use when troubleshooting complex issues
            - Includes: pod status, logs, image analysis, network tests

  summary   - Quick diagnostic with actionable recommendations
            - Use for regular health checks
            - Provides prioritized action plan

  pod       - Analyze pod status, restart counts, logs
            - Check container health and recent events

  image     - Verify Docker image integrity
            - Check if binary exists and is executable

  source    - Analyze source code for available endpoints
            - Verify route definitions and configurations

  network   - Test internal and external connectivity
            - Verify database connections and API access

EXAMPLES:
━━━━━━━━━
  $(basename "$0") debug summary         # Quick health check
  $(basename "$0") debug full           # Complete analysis
  $(basename "$0") debug pod            # Check pod issues
EOM
}

function show_test_help() {
    cat <<EOM
🧪 TESTING COMMANDS
===================

Usage: $(basename "$0") test <command>

COMMANDS:
━━━━━━━━━
  api              Quick API endpoint test (recommended)
  endpoints        Detailed API endpoint analysis
  clean            Clean, standalone API test suite

COMMAND DESCRIPTIONS:
━━━━━━━━━━━━━━━━━━━━━
  api         - Fast API testing for regular use
              - Tests all core endpoints
              - Shows semantic feature status

  endpoints   - Detailed endpoint testing with analysis
              - Part of full debugging workflow
              - Includes error analysis

  clean       - Standalone test suite
              - Clean output format
              - Good for CI/CD integration

EXAMPLES:
━━━━━━━━━
  $(basename "$0") test api             # Quick API test
  $(basename "$0") test clean           # Clean test output
  $(basename "$0") test endpoints       # Detailed analysis
EOM
}

function show_category_help() {
    local category="$1"
    case "$category" in
        secrets) show_secrets_help ;;
        debug) show_debug_help ;;
        test) show_test_help ;;
        *) 
            echo "❌ Unknown category: $category"
            echo "Available categories: secrets, debug, test"
            show_main_usage
            ;;
    esac
}

# ─────────────────────────────────────────────────────────────────────────────
# 🛠️ UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
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
    local dashes=""
    for ((i=0; i<$((${#title} + 4)); i++)); do
        dashes+="-"
    done
    echo "$dashes"
}

function ensure_namespace_exists() {
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        echo "📁 Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
}

function check_deployment_prerequisites() {
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

function get_latest_pod_info() {
    local pods=$(kubectl get pods -n "$NAMESPACE" -l "$APP_LABEL" -o json)
    POD=$(echo "$pods" | jq -r '.items | sort_by(.metadata.creationTimestamp) | reverse | .[0].metadata.name')
    
    if [[ -z "${POD:-}" || "$POD" == "null" ]]; then
        echo "❌ Failed to get pod name"
        exit 1
    fi
    
    echo "🎯 Selected pod: $POD"
}

# ─────────────────────────────────────────────────────────────────────────────
# 🔐 SECRETS MANAGEMENT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
function load_environment_file() {
    local env_file="${1:-.env}"
    
    if [[ ! -f "$env_file" ]]; then
        echo "❌ Environment file '$env_file' not found"
        echo "💡 Create a .env file with:"
        echo "   OPENAI_API_KEY=sk-your-real-key-here"
        echo "   DATABASE_URL=postgres://cache:cache@postgres.infra:5432/cache?sslmode=disable"
        return 1
    fi
    
    echo "📄 Loading environment variables from $env_file..."
    
    # Load .env file safely
    set -a  # automatically export all variables
    source "$env_file"
    set +a  # disable auto-export
    
    echo "✅ Environment variables loaded"
    echo "🔑 OPENAI_API_KEY: ${OPENAI_API_KEY:0:8}... (${#OPENAI_API_KEY} chars)"
    echo "🗄️ DATABASE_URL: ${DATABASE_URL:0:20}... (${#DATABASE_URL} chars)"
}

function get_openai_api_key() {
    local openai_key="${OPENAI_API_KEY:-}"
    
    # If no key in environment, try to prompt (only in interactive mode)
    if [[ -z "$openai_key" ]]; then
        if [[ -t 0 ]]; then  # Check if running interactively
            echo "🔑 OpenAI API key not found in environment variables."
            echo "Enter your OpenAI API key (or press Enter to use dummy-key):"
            read -s openai_key
            echo ""
        else
            echo "⚠️  No OPENAI_API_KEY found in environment and not running interactively."
            echo "Using dummy-key. Semantic features will be disabled."
            openai_key="dummy-key"
        fi
    fi
    
    # Default to dummy-key if still empty
    if [[ -z "$openai_key" ]]; then
        echo "⚠️  No API key provided. Using dummy-key. Semantic features will be disabled."
        openai_key="dummy-key"
    fi
    
    # Validate key format (basic check)
    if [[ "$openai_key" != "dummy-key" && ! "$openai_key" =~ ^sk- ]]; then
        echo "⚠️  Warning: OpenAI API key should start with 'sk-'. Current key: ${openai_key:0:8}..."
        echo "Are you sure this is correct? (y/n)"
        if [[ -t 0 ]]; then
            read -n 1 confirm
            echo ""
            if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
                echo "❌ Aborted."
                exit 1
            fi
        fi
    fi
    
    echo "$openai_key"
}

function create_kubernetes_secrets() {
    echo "🔐 Creating Kubernetes secrets..."
    
    ensure_namespace_exists
    
    # Get OpenAI API key from environment or prompt
    local openai_key
    openai_key=$(get_openai_api_key)
    
    # Default database URL (can be overridden by environment)
    local db_url="${DATABASE_URL:-postgres://cache:cache@postgres.infra:5432/cache?sslmode=disable}"
    
    echo "📝 Creating secret: $SECRET_NAME"
    echo "🔑 OpenAI Key: ${openai_key:0:8}... (${#openai_key} characters)"
    echo "🗄️ Database: ${db_url%%\?*} (connection string)"
    
    # Delete existing secret if it exists
    kubectl delete secret "$SECRET_NAME" -n "$NAMESPACE" 2>/dev/null || true
    
    # Create new secret
    kubectl create secret generic "$SECRET_NAME" \
        --namespace="$NAMESPACE" \
        --from-literal=DATABASE_URL="$db_url" \
        --from-literal=OPENAI_API_KEY="$openai_key"
    
    echo "✅ Secret created successfully!"
    
    # Show status
    if [[ "$openai_key" == "dummy-key" ]]; then
        echo "⚠️  Using dummy OpenAI key - semantic features will be disabled"
        echo "💡 To enable semantic features later:"
        echo "   export OPENAI_API_KEY='sk-your-real-key'"
        echo "   $(basename "$0") secrets update-openai"
    else
        echo "🎉 Real OpenAI key detected - semantic features will be enabled!"
    fi
    
    echo ""
    echo "🔍 To verify:"
    echo "  kubectl get secret $SECRET_NAME -n $NAMESPACE"
    echo "  $(basename "$0") secrets view"
    echo ""
    echo "🚀 Next steps:"
    echo "  1. Apply the secure deployment manifest"
    echo "  2. kubectl apply -f secure-sematic-cache.yaml"
}

function update_openai_api_key() {
    echo "🔑 Updating OpenAI API key..."
    
    # Check if key is in environment first
    local new_key="${OPENAI_API_KEY:-}"
    
    if [[ -n "$new_key" ]]; then
        echo "🌍 Using OpenAI key from environment variable"
        echo "🔑 Key: ${new_key:0:8}... (${#new_key} characters)"
    else
        if [[ -t 0 ]]; then
            echo "Enter new OpenAI API key:"
            read -s new_key
            echo ""
        else
            echo "❌ No OPENAI_API_KEY in environment and not running interactively"
            echo "💡 Set environment variable: export OPENAI_API_KEY='sk-your-key'"
            exit 1
        fi
    fi
    
    if [[ -z "$new_key" ]]; then
        echo "❌ Empty key provided. Aborted."
        exit 1
    fi
    
    # Validate key format
    if [[ "$new_key" != "dummy-key" && ! "$new_key" =~ ^sk- ]]; then
        echo "⚠️  Warning: OpenAI API key should start with 'sk-'"
    fi
    
    # Update the secret
    kubectl patch secret "$SECRET_NAME" -n "$NAMESPACE" -p="{\"data\":{\"OPENAI_API_KEY\":\"$(echo -n "$new_key" | base64)\"}}"
    
    echo "✅ OpenAI API key updated!"
    echo "🔄 Restart deployment to pick up new key:"
    echo "  kubectl rollout restart deployment/sematic-cache -n $NAMESPACE"
}

function update_database_url() {
    echo "🗄️ Updating database URL..."
    
    # Check if URL is in environment first
    local new_url="${DATABASE_URL:-}"
    
    if [[ -n "$new_url" ]]; then
        echo "🌍 Using database URL from environment variable"
        echo "🗄️ URL: ${new_url%%\?*}... (connection string)"
    else
        if [[ -t 0 ]]; then
            echo "Enter new database URL:"
            read new_url
            echo ""
        else
            echo "❌ No DATABASE_URL in environment and not running interactively"
            exit 1
        fi
    fi
    
    if [[ -z "$new_url" ]]; then
        echo "❌ Empty URL provided. Aborted."
        exit 1
    fi
    
    # Update the secret
    kubectl patch secret "$SECRET_NAME" -n "$NAMESPACE" -p="{\"data\":{\"DATABASE_URL\":\"$(echo -n "$new_url" | base64)\"}}"
    
    echo "✅ Database URL updated!"
    echo "🔄 Restart deployment to pick up new URL:"
    echo "  kubectl rollout restart deployment/sematic-cache -n $NAMESPACE"
}

function view_current_secrets() {
    echo "🔍 Current secrets (decoded):"
    echo "=============================="
    
    if ! kubectl get secret "$SECRET_NAME" -n "$NAMESPACE" >/dev/null 2>&1; then
        echo "❌ Secret '$SECRET_NAME' not found in namespace '$NAMESPACE'"
        echo "💡 Run: $(basename "$0") secrets create"
        return 1
    fi
    
    echo "🔑 OPENAI_API_KEY:"
    local openai_key=$(kubectl get secret "$SECRET_NAME" -n "$NAMESPACE" -o jsonpath='{.data.OPENAI_API_KEY}' | base64 --decode)
    echo "   ${openai_key:0:8}... (${#openai_key} characters)"
    if [[ "$openai_key" == "dummy-key" ]]; then
        echo "   ⚠️  Using dummy key - semantic features disabled"
    else
        echo "   ✅ Real API key - semantic features enabled"
    fi
    echo ""
    
    echo "🗄️ DATABASE_URL:"
    local db_url=$(kubectl get secret "$SECRET_NAME" -n "$NAMESPACE" -o jsonpath='{.data.DATABASE_URL}' | base64 --decode)
    echo "   ${db_url%%\?*}... (connection string)"
    echo ""
    
    echo "📊 Secret metadata:"
    kubectl get secret "$SECRET_NAME" -n "$NAMESPACE" -o yaml | head -20
}

function delete_all_secrets() {
    echo "🗑️ Deleting secrets..."
    
    echo "⚠️  This will delete all secrets for sematic-cache!"
    echo "Are you sure? (yes/no)"
    read confirmation
    
    if [[ "$confirmation" != "yes" ]]; then
        echo "❌ Aborted."
        return 1
    fi
    
    kubectl delete secret "$SECRET_NAME" -n "$NAMESPACE" 2>/dev/null || echo "Secret not found"
    kubectl delete configmap "$CONFIGMAP_NAME" -n "$NAMESPACE" 2>/dev/null || echo "ConfigMap not found"
    
    echo "✅ Secrets deleted."
}

function create_configuration_map() {
    echo "⚙️ Creating non-sensitive configuration..."
    
    ensure_namespace_exists
    
    # Delete existing configmap if it exists
    kubectl delete configmap "$CONFIGMAP_NAME" -n "$NAMESPACE" 2>/dev/null || true
    
    # Create configmap
    kubectl create configmap "$CONFIGMAP_NAME" \
        --namespace="$NAMESPACE" \
        --from-literal=GIN_MODE="release" \
        --from-literal=LOG_LEVEL="info" \
        --from-literal=SERVER_ADDRESS=":8080"
    
    echo "✅ ConfigMap created successfully!"
}

function create_sample_environment_file() {
    echo "📄 Creating sample .env file..."
    
    cat > .env.example << 'EOF'
# Sematic Cache Environment Variables
# Copy this file to .env and fill in your actual values

# OpenAI API Key (get from https://platform.openai.com/)
OPENAI_API_KEY=sk-your-real-openai-api-key-here

# Database URL (usually you don't need to change this for local k3d)
DATABASE_URL=postgres://cache:cache@postgres.infra:5432/cache?sslmode=disable

# Optional: Override other settings
# GIN_MODE=release
# LOG_LEVEL=info
EOF

    echo "✅ Created .env.example"
    echo "💡 Copy to .env and fill in your values:"
    echo "   cp .env.example .env"
    echo "   # Edit .env with your actual OpenAI API key"
}

# ─────────────────────────────────────────────────────────────────────────────
# 🧪 API TESTING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
function test_single_endpoint() {
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

function test_endpoint_clean_format() {
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

function run_clean_api_tests() {
    echo "🚀 Sematic Cache API Test Suite"
    echo "==============================="
    
    echo ""
    echo "⚠️  OpenAI API Key Status: Using 'dummy-key'"
    echo "• ✅ Basic storage works"  
    echo "• ❌ Semantic search requires real OpenAI key"
    
    # Core functionality tests
    test_endpoint_clean_format "📊 Health Check" "GET" "/health" ""
    test_endpoint_clean_format "📈 Metrics" "GET" "/metrics" ""
    test_endpoint_clean_format "💾 Cache SET" "POST" "/set" '{"key": "test", "value": "value"}'
    test_endpoint_clean_format "🔍 Cache GET" "POST" "/get" '{"key": "test"}'
    test_endpoint_clean_format "🔎 Semantic Query" "POST" "/query" '{"query": "test", "threshold": 0.8}'
    test_endpoint_clean_format "🏆 Top-K Search" "POST" "/topk" '{"query": "test", "k": 5}'
    test_endpoint_clean_format "🧹 Admin Flush" "POST" "/admin/flush" '{}'
    
    echo ""
    echo ""
    echo "✅ Testing Complete!"
    echo ""
    echo "🌐 Access your API at: http://sematic.127.0.0.1.nip.io:8080"
    echo ""
    echo "💡 To enable semantic features:"
    echo "   Get an OpenAI API key and update the deployment"
    echo "   $(basename "$0") secrets update-openai"
}

function test_api_endpoints_detailed() {
    log_section "🚀 API Endpoint Testing"
    
    echo ""
    echo "⚠️  IMPORTANT: OpenAI API Key Status"
    echo "Your app is using 'dummy-key' for OpenAI API, which means:"
    echo "• ✅ Basic storage works (data is stored)"  
    echo "• ❌ Semantic search disabled (no embeddings)"
    echo "• ❌ GET operations return empty (can't find without embeddings)"
    
    # Core endpoints
    test_single_endpoint "Health Check" "GET" "/health" ""
    test_single_endpoint "Metrics" "GET" "/metrics" ""
    
    # Storage operations
    test_single_endpoint "Cache SET Operation" "POST" "/set" \
      '{"key": "test-basic", "value": "basic storage test"}'
    test_single_endpoint "Cache GET Operation" "POST" "/get" \
      '{"key": "test-basic"}'
    
    # AI-dependent features  
    test_single_endpoint "Query Operation (semantic search)" "POST" "/query" \
      '{"query": "test question", "threshold": 0.8}'
    test_single_endpoint "Top-K Search (vector search)" "POST" "/topk" \
      '{"query": "test", "k": 5}'
    
    # Admin operations
    test_single_endpoint "Admin Flush" "POST" "/admin/flush" '{}'
}

function run_quick_api_tests() {
    echo "🧪 Quick API Test"
    echo "=================="
    
    # Test key endpoints quickly
    local health_status=$(curl -s -o /dev/null -w "%{http_code}" -H "Host: $HOST_HEADER" http://$BASE_URL/health 2>/dev/null || echo "000")
    local set_status=$(curl -s -o /dev/null -w "%{http_code}" -H "Host: $HOST_HEADER" -H "Content-Type: application/json" -X POST http://$BASE_URL/set -d '{"key":"test","value":"test"}' 2>/dev/null || echo "000")
    local query_status=$(curl -s -o /dev/null -w "%{http_code}" -H "Host: $HOST_HEADER" -H "Content-Type: application/json" -X POST http://$BASE_URL/query -d '{"query":"test"}' 2>/dev/null || echo "000")
    
    echo "📊 Health Endpoint: $([[ "$health_status" == "200" ]] && echo "✅ $health_status" || echo "❌ $health_status")"
    echo "💾 Storage Endpoint: $([[ "$set_status" =~ ^(200|201)$ ]] && echo "✅ $set_status" || echo "❌ $set_status")"
    echo "🔎 Query Endpoint: $([[ "$query_status" == "200" ]] && echo "✅ $query_status" || echo "❌ $query_status")"
    
    echo ""
    if [[ "$health_status" == "200" && "$set_status" =~ ^(200|201)$ ]]; then
        echo "🎉 Core functionality is working!"
        if [[ "$query_status" != "200" ]]; then
            echo "⚠️  Semantic features need attention (query endpoint issue)"
        fi
    else
        echo "❌ Core functionality has issues - run '$(basename "$0") debug summary' for analysis"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 🐳 POD AND DEPLOYMENT ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
function analyze_pod_detailed_status() {
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

function show_pod_logs_and_events() {
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
    
    log_section "📅 Recent Kubernetes Events"
    kubectl get events -n "$NAMESPACE" --field-selector involvedObject.name="$POD" --sort-by='.lastTimestamp' | tail -10
}

function analyze_docker_image_integrity() {
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

function analyze_source_code_structure() {
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

# ─────────────────────────────────────────────────────────────────────────────
# 📋 COMPREHENSIVE ISSUE ANALYSIS AND SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
function run_comprehensive_issue_analysis() {
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
    echo "   $(basename "$0") secrets update-openai"
    
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

# ─────────────────────────────────────────────────────────────────────────────
# 🧭 COMMAND DISPATCHERS
# ─────────────────────────────────────────────────────────────────────────────
function handle_secrets_commands() {
    local command="${1:-help}"
    
    case "$command" in
        create)
            create_kubernetes_secrets
            ;;
        update-openai)
            update_openai_api_key
            ;;
        update-db)
            update_database_url
            ;;
        view)
            view_current_secrets
            ;;
        delete)
            delete_all_secrets
            ;;
        create-config)
            create_configuration_map
            ;;
        load-env)
            load_environment_file "${2:-.env}"
            ;;
        create-env)
            create_sample_environment_file
            ;;
        help|*)
            show_secrets_help
            ;;
    esac
}

function handle_debug_commands() {
    local command="${1:-help}"
    
    case "$command" in
        full)
            echo "🔍 COMPREHENSIVE SEMATIC CACHE DEBUGGING"
            echo "========================================"
            check_deployment_prerequisites
            get_latest_pod_info
            analyze_pod_detailed_status
            show_pod_logs_and_events
            test_api_endpoints_detailed
            analyze_docker_image_integrity
            analyze_source_code_structure
            test_network_connectivity
            run_comprehensive_issue_analysis
            ;;
        summary)
            echo "📋 ISSUE ANALYSIS & SUMMARY"
            echo "==========================="
            check_deployment_prerequisites
            get_latest_pod_info
            run_comprehensive_issue_analysis
            ;;
        pod)
            echo "📦 POD STATUS ANALYSIS"
            echo "====================="
            check_deployment_prerequisites
            get_latest_pod_info
            analyze_pod_detailed_status
            show_pod_logs_and_events
            ;;
        image)
            echo "🐳 DOCKER IMAGE ANALYSIS"
            echo "========================"
            check_deployment_prerequisites
            get_latest_pod_info
            analyze_docker_image_integrity
            ;;
        source)
            echo "📄 SOURCE CODE ANALYSIS"
            echo "======================"
            analyze_source_code_structure
            ;;
        network)
            echo "🌐 NETWORK CONNECTIVITY TESTING"
            echo "==============================="
            check_deployment_prerequisites
            get_latest_pod_info
            test_network_connectivity
            ;;
        help|*)
            show_debug_help
            ;;
    esac
}

function handle_test_commands() {
    local command="${1:-help}"
    
    case "$command" in
        api)
            run_quick_api_tests
            ;;
        endpoints)
            check_deployment_prerequisites
            get_latest_pod_info
            test_api_endpoints_detailed
            ;;
        clean)
            run_clean_api_tests
            ;;
        help|*)
            show_test_help
            ;;
    esac
}

# ─────────────────────────────────────────────────────────────────────────────
# 🚀 MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
function main() {
    local category="${1:-help}"
    local command="${2:-help}"
    
    case "$category" in
        secrets)
            handle_secrets_commands "$command" "$3"
            ;;
        debug)
            handle_debug_commands "$command"
            ;;
        test)
            handle_test_commands "$command"
            ;;
        help)
            if [[ -n "$command" && "$command" != "help" ]]; then
                show_category_help "$command"
            else
                show_main_usage
            fi
            ;;
        *)
            echo "❌ Unknown category: $category"
            echo ""
            show_main_usage
            ;;
    esac
}

# Entry point - only run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi