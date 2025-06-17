#!/usr/bin/env bash
# =====================================
# 🧪 END-TO-END TEST ORCHESTRATOR
# =====================================

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source configuration and framework
source "$SCRIPT_DIR/lib/config.sh"
source "$SCRIPT_DIR/lib/test-framework.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [TEST_SUITE...]

Production-ready end-to-end testing for semantic cache deployment.

TEST_SUITES:
  all                Run all test suites (default)
  infrastructure     Test cluster, namespaces, and infrastructure components
  application        Test application deployment and services
  api                Test API endpoints and functionality
  performance        Test performance and load handling
  monitoring         Test logging and observability

OPTIONS:
  -h, --help         Show this help message
  -t, --timeout N    Set timeout for health checks (default: 300s)
  -r, --retries N    Set retry count for health checks (default: 30)
  -v, --verbose      Enable verbose output
  --base-url URL     Set base URL for API tests (default: http://localhost:8080)
  --quick           Skip performance and monitoring tests for faster execution

EXAMPLES:
  $(basename "$0")                         # Run all tests
  $(basename "$0") infrastructure          # Test only infrastructure
  $(basename "$0") api performance         # Test API and performance
  $(basename "$0") --quick all            # Run essential tests only
  $(basename "$0") --timeout 600 all      # Run all tests with 10min timeout

COMPREHENSIVE WORKFLOW:
  # Complete end-to-end testing workflow
  deploy/cluster.sh up                     # 1. Create cluster
  deploy/dev.sh build                      # 2. Build application
  deploy/dev.sh deploy                     # 3. Deploy application
  deploy/tests/e2e.sh all                  # 4. Run all tests
  deploy/cluster.sh down                   # 5. Cleanup (optional)

EOF
}

load_test_suite() {
    local suite="$1"
    local suite_file="$SCRIPT_DIR/suites/${suite}.sh"
    
    if [ -f "$suite_file" ]; then
        source "$suite_file"
        success "Loaded $suite test suite"
    else
        error "Test suite '$suite' not found at $suite_file"
        return 1
    fi
}

run_suite() {
    local suite="$1"
    
    case "$suite" in
        infrastructure)
            load_test_suite "infrastructure"
            run_test_suite "Infrastructure" \
                test_cluster_creation \
                test_namespaces \
                test_infrastructure_components
            ;;
        application)
            load_test_suite "application"
            run_test_suite "Application" \
                test_application_deployments \
                test_services \
                test_ingress \
                test_config_maps \
                test_secrets
            ;;
        api)
            load_test_suite "api"
            run_test_suite "API" \
                test_health_endpoints \
                test_web_interface \
                test_cache_operations \
                test_advanced_operations \
                test_error_handling \
                test_concurrent_requests
            ;;
        performance)
            load_test_suite "performance"
            run_test_suite "Performance" \
                test_response_time \
                test_concurrent_performance \
                test_cache_performance \
                test_memory_usage \
                test_cpu_usage \
                test_load_handling
            ;;
        monitoring)
            load_test_suite "monitoring"
            run_test_suite "Monitoring" \
                test_application_logging \
                test_infrastructure_logging \
                test_resource_metrics \
                test_pod_health_checks \
                test_restart_counts \
                test_network_connectivity
            ;;
        *)
            error "Unknown test suite: $suite"
            return 1
            ;;
    esac
}

main() {
    local test_suites=()
    local quick_mode=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -t|--timeout)
                TEST_TIMEOUT="$2"
                shift 2
                ;;
            -r|--retries)
                HEALTH_CHECK_RETRIES="$2"
                shift 2
                ;;
            --base-url)
                BASE_URL="$2"
                shift 2
                ;;
            -v|--verbose)
                set -x
                shift
                ;;
            --quick)
                quick_mode=true
                shift
                ;;
            infrastructure|application|api|performance|monitoring)
                test_suites+=("$1")
                shift
                ;;
            all)
                if [ "$quick_mode" = true ]; then
                    test_suites=(infrastructure application api)
                else
                    test_suites=(infrastructure application api performance monitoring)
                fi
                shift
                ;;
            *)
                error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Default to all tests if none specified
    if [ ${#test_suites[@]} -eq 0 ]; then
        if [ "$quick_mode" = true ]; then
            test_suites=(infrastructure application api)
        else
            test_suites=(infrastructure application api performance monitoring)
        fi
    fi
    
    # Header
    log "🚀 Starting end-to-end tests for semantic cache deployment"
    log "Cluster: $CLUSTER_NAME"
    log "Base URL: $BASE_URL"
    log "Test Suites: ${test_suites[*]}"
    log "Quick Mode: $quick_mode"
    echo
    
    # Check prerequisites
    if ! check_prerequisites; then
        exit 1
    fi
    
    # Verify cluster is accessible
    if ! kubectl --context "$KUBE_CONTEXT" cluster-info >/dev/null 2>&1; then
        error "Cannot connect to cluster '$CLUSTER_NAME'"
        log "💡 Try running: deploy/cluster.sh up"
        exit 1
    fi
    
    # Run test suites
    local suite_failures=0
    for suite in "${test_suites[@]}"; do
        log "🎯 Starting $suite test suite..."
        echo "=================================="
        
        local suite_start_time=$(date +%s)
        
        if run_suite "$suite"; then
            local suite_end_time=$(date +%s)
            local suite_duration=$((suite_end_time - suite_start_time))
            success "✅ $suite test suite completed in ${suite_duration}s"
        else
            error "❌ $suite test suite failed"
            ((suite_failures++))
        fi
        echo
    done
    
    # Final summary
    log "🏁 All test suites completed"
    if [ $suite_failures -gt 0 ]; then
        error "$suite_failures test suite(s) had failures"
    fi
    
    show_summary
    
    # Exit with appropriate code
    if [ $TESTS_FAILED -gt 0 ] || [ $suite_failures -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi