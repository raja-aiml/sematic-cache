#!/bin/bash
set -euo pipefail

# Test All Script
# Runs all tests for the K3D Blueprint project

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🧪 Testing K3D Blueprint"
echo "======================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if cluster exists
if ! k3d cluster list | grep -q "local-k8s"; then
    echo -e "${RED}❌ K3D cluster 'local-k8s' not found${NC}"
    echo "Please create the cluster first: k3d cluster create local-k8s"
    exit 1
fi

# Set kubeconfig
export KUBECONFIG=$(k3d kubeconfig write local-k8s)

# Run smoke tests
echo -e "\n💨 Running Smoke Tests:"
echo "---------------------"

if [ -f "$PROJECT_ROOT/validation-kit/scripts/smoke-test.sh" ]; then
    bash "$PROJECT_ROOT/validation-kit/scripts/smoke-test.sh"
else
    echo -e "${YELLOW}⚠${NC} Smoke test script not found"
fi

# Test base infrastructure
echo -e "\n🏗️  Testing Base Infrastructure:"
echo "------------------------------"

# Deploy minimal scenario
echo "Deploying minimal scenario..."
kubectl apply -k "$PROJECT_ROOT/scenarios/minimal" --dry-run=client > /dev/null 2>&1 && echo -e "${GREEN}✓${NC} Minimal scenario valid" || echo -e "${RED}✗${NC} Minimal scenario invalid"

# Test PostgreSQL
echo -n "Testing PostgreSQL deployment... "
kubectl apply -f "$PROJECT_ROOT/infra/base/postgres/deployment.yaml" --dry-run=client > /dev/null 2>&1 && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"

# Test Redis
echo -n "Testing Redis deployment... "
kubectl apply -f "$PROJECT_ROOT/infra/base/redis/deployment.yaml" --dry-run=client > /dev/null 2>&1 && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"

# Test modules
echo -e "\n🧩 Testing Modules:"
echo "------------------"

# Test Istio module
echo -n "Testing Istio module... "
kubectl apply -k "$PROJECT_ROOT/infra/modules/istio" --dry-run=client > /dev/null 2>&1 && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"

# Test Observability module
echo -n "Testing Observability module... "
kubectl apply -k "$PROJECT_ROOT/infra/modules/observability" --dry-run=client > /dev/null 2>&1 && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"

# Test Security module
echo -n "Testing Security module... "
kubectl apply -k "$PROJECT_ROOT/infra/modules/security" --dry-run=client > /dev/null 2>&1 && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"

# Test overlays
echo -e "\n🔀 Testing Overlays:"
echo "-------------------"

for overlay in local dev; do
    echo -n "Testing $overlay overlay... "
    kubectl apply -k "$PROJECT_ROOT/infra/overlays/$overlay" --dry-run=client > /dev/null 2>&1 && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"
done

# Test scenarios
echo -e "\n🎯 Testing Scenarios:"
echo "--------------------"

for scenario in minimal development service-mesh monitoring-only full-stack; do
    echo -n "Testing $scenario scenario... "
    kubectl apply -k "$PROJECT_ROOT/scenarios/$scenario" --dry-run=client > /dev/null 2>&1 && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"
done

# Integration tests
echo -e "\n🔗 Running Integration Tests:"
echo "----------------------------"

# Deploy minimal scenario for testing
echo "Deploying minimal scenario for integration tests..."
kubectl apply -k "$PROJECT_ROOT/scenarios/minimal" 2>/dev/null

# Wait for deployments
echo "Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/postgres -n infra 2>/dev/null || echo -e "${YELLOW}⚠${NC} PostgreSQL not ready"
kubectl wait --for=condition=available --timeout=300s deployment/redis -n infra 2>/dev/null || echo -e "${YELLOW}⚠${NC} Redis not ready"

# Test connectivity
echo -e "\n🔌 Testing Connectivity:"
echo "-----------------------"

# Test PostgreSQL connectivity
echo -n "PostgreSQL connectivity... "
if kubectl run postgres-test --image=postgres:16-alpine --rm -it --restart=Never -n infra -- psql -h postgres -U postgres -c "SELECT 1" 2>/dev/null | grep -q "1 row"; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
fi

# Test Redis connectivity
echo -n "Redis connectivity... "
if kubectl run redis-test --image=redis:7-alpine --rm -it --restart=Never -n infra -- redis-cli -h redis ping 2>/dev/null | grep -q "PONG"; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
fi

# Performance tests
echo -e "\n⚡ Running Performance Tests:"
echo "----------------------------"

# Test PostgreSQL performance
echo "PostgreSQL benchmark (quick):"
kubectl run pgbench-test --image=postgres:16-alpine --rm -it --restart=Never -n infra -- pgbench -h postgres -U postgres -d postgres -T 10 -c 5 2>/dev/null || echo -e "${YELLOW}⚠${NC} pgbench test skipped"

# Test Redis performance
echo "Redis benchmark (quick):"
kubectl run redis-bench-test --image=redis:7-alpine --rm -it --restart=Never -n infra -- redis-benchmark -h redis -t set,get -n 1000 -q 2>/dev/null || echo -e "${YELLOW}⚠${NC} redis-benchmark test skipped"

# Cleanup
echo -e "\n🧹 Cleaning up test resources..."
kubectl delete -k "$PROJECT_ROOT/scenarios/minimal" 2>/dev/null || true

echo -e "\n✅ All tests completed!"