#!/bin/bash
set -euo pipefail

# Integration Test Script
# Runs integration tests for K3D Blueprint components

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATION_KIT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔗 Running Integration Tests"
echo "==========================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Test namespace
NAMESPACE="integration-test"

# Create test namespace
echo "Creating test namespace..."
kubectl create namespace $NAMESPACE 2>/dev/null || true

# Function to run integration test
run_integration_test() {
    local name=$1
    local file=$2
    
    echo -n "Running $name... "
    
    if [ -f "$file" ]; then
        if kubectl apply -f "$file" > /dev/null 2>&1; then
            # Wait for job to complete
            job_name=$(basename "$file" .yaml | sed 's/_/-/g')
            if kubectl wait --for=condition=complete job/$job_name -n $NAMESPACE --timeout=300s > /dev/null 2>&1; then
                echo -e "${GREEN}✓ PASS${NC}"
                # Show job logs
                kubectl logs job/$job_name -n $NAMESPACE 2>/dev/null | sed 's/^/  /'
                return 0
            else
                echo -e "${RED}✗ FAIL (timeout or error)${NC}"
                # Show job logs if available
                kubectl logs job/$job_name -n $NAMESPACE 2>/dev/null | tail -20 | sed 's/^/  /'
                return 1
            fi
        else
            echo -e "${RED}✗ FAIL (failed to create job)${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠ SKIP (test not found)${NC}"
        return 0
    fi
}

# Run integration tests
echo -e "\n📋 Component Integration Tests:"
echo "------------------------------"

TOTAL_PASS=0
TOTAL_FAIL=0

# PostgreSQL integration test
if run_integration_test "PostgreSQL Integration" "$VALIDATION_KIT_ROOT/tests/integration/postgres-test.yaml"; then
    ((TOTAL_PASS++))
else
    ((TOTAL_FAIL++))
fi

# Redis integration test
if run_integration_test "Redis Integration" "$VALIDATION_KIT_ROOT/tests/integration/redis-test.yaml"; then
    ((TOTAL_PASS++))
else
    ((TOTAL_FAIL++))
fi

# Ingress integration test
if run_integration_test "Ingress Integration" "$VALIDATION_KIT_ROOT/tests/integration/ingress-test.yaml"; then
    ((TOTAL_PASS++))
else
    ((TOTAL_FAIL++))
fi

# Istio integration test (if available)
if [ -f "$VALIDATION_KIT_ROOT/tests/integration/istio-test.yaml" ]; then
    if run_integration_test "Istio Integration" "$VALIDATION_KIT_ROOT/tests/integration/istio-test.yaml"; then
        ((TOTAL_PASS++))
    else
        ((TOTAL_FAIL++))
    fi
fi

# Monitoring integration test (if available)
if [ -f "$VALIDATION_KIT_ROOT/tests/integration/monitoring-test.yaml" ]; then
    if run_integration_test "Monitoring Integration" "$VALIDATION_KIT_ROOT/tests/integration/monitoring-test.yaml"; then
        ((TOTAL_PASS++))
    else
        ((TOTAL_FAIL++))
    fi
fi

# Tracing integration test (if available)
if [ -f "$VALIDATION_KIT_ROOT/tests/integration/tracing-test.yaml" ]; then
    if run_integration_test "Tracing Integration" "$VALIDATION_KIT_ROOT/tests/integration/tracing-test.yaml"; then
        ((TOTAL_PASS++))
    else
        ((TOTAL_FAIL++))
    fi
fi

# Summary
echo -e "\n📊 Integration Test Summary"
echo "=========================="
echo -e "Passed: ${GREEN}$TOTAL_PASS${NC}"
echo -e "Failed: ${RED}$TOTAL_FAIL${NC}"

# Cleanup
echo -e "\n🧹 Cleaning up..."
kubectl delete namespace $NAMESPACE --ignore-not-found=true 2>/dev/null || true

# Exit code
if [ $TOTAL_FAIL -gt 0 ]; then
    exit 1
else
    exit 0
fi