#!/bin/bash
set -euo pipefail

# Run All Tests Script
# Executes the complete test suite for K3D Blueprint

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATION_KIT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🧪 K3D Blueprint Test Suite"
echo "=========================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Test results
PASSED=0
FAILED=0
SKIPPED=0

# Run test and track results
run_test() {
    local name=$1
    local script=$2
    
    echo -e "\n📋 Running $name..."
    echo "----------------------------------------"
    
    if [ -f "$script" ]; then
        if bash "$script"; then
            echo -e "${GREEN}✓ $name passed${NC}"
            ((PASSED++))
        else
            echo -e "${RED}✗ $name failed${NC}"
            ((FAILED++))
        fi
    else
        echo -e "${YELLOW}⚠ $name skipped (script not found)${NC}"
        ((SKIPPED++))
    fi
}

# Check prerequisites
echo "Checking prerequisites..."
if ! kubectl get nodes > /dev/null 2>&1; then
    echo -e "${RED}❌ Kubernetes cluster not accessible${NC}"
    exit 1
fi

# Run test suites
run_test "Smoke Tests" "$SCRIPT_DIR/smoke-test.sh"
run_test "Integration Tests" "$SCRIPT_DIR/integration-test.sh"
run_test "Performance Tests" "$SCRIPT_DIR/performance-test.sh"
run_test "Security Tests" "$SCRIPT_DIR/security-test.sh"
run_test "Monitoring Tests" "$SCRIPT_DIR/monitoring-test.sh"

# Generate report
echo -e "\n📊 Test Summary"
echo "==============="
echo -e "Passed:  ${GREEN}$PASSED${NC}"
echo -e "Failed:  ${RED}$FAILED${NC}"
echo -e "Skipped: ${YELLOW}$SKIPPED${NC}"

# Generate JUnit report for CI
cat > "$VALIDATION_KIT_ROOT/test-results.xml" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="K3D Blueprint Tests" tests="$((PASSED + FAILED + SKIPPED))" failures="$FAILED" skipped="$SKIPPED">
  <testsuite name="Validation Kit" tests="$((PASSED + FAILED + SKIPPED))" failures="$FAILED" skipped="$SKIPPED">
    <testcase classname="SmokeTests" name="smoke-test" time="0">
      $([ -f "$SCRIPT_DIR/smoke-test.sh" ] && echo "" || echo '<skipped/>')
    </testcase>
    <testcase classname="IntegrationTests" name="integration-test" time="0">
      $([ -f "$SCRIPT_DIR/integration-test.sh" ] && echo "" || echo '<skipped/>')
    </testcase>
    <testcase classname="PerformanceTests" name="performance-test" time="0">
      $([ -f "$SCRIPT_DIR/performance-test.sh" ] && echo "" || echo '<skipped/>')
    </testcase>
    <testcase classname="SecurityTests" name="security-test" time="0">
      $([ -f "$SCRIPT_DIR/security-test.sh" ] && echo "" || echo '<skipped/>')
    </testcase>
    <testcase classname="MonitoringTests" name="monitoring-test" time="0">
      $([ -f "$SCRIPT_DIR/monitoring-test.sh" ] && echo "" || echo '<skipped/>')
    </testcase>
  </testsuite>
</testsuites>
EOF

# Cleanup test namespaces
echo -e "\n🧹 Cleaning up test resources..."
kubectl delete namespace integration-test performance-test security-test --ignore-not-found=true 2>/dev/null || true

# Exit with appropriate code
if [ $FAILED -gt 0 ]; then
    echo -e "\n${RED}❌ Test suite failed${NC}"
    exit 1
else
    echo -e "\n${GREEN}✅ Test suite passed${NC}"
    exit 0
fi