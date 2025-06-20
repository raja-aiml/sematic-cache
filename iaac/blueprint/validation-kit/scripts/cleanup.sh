#!/bin/bash
set -euo pipefail

# Cleanup Script
# Removes all test resources from the cluster

echo "🧹 Cleaning up test resources"
echo "============================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Test namespaces to clean
TEST_NAMESPACES=(
    "integration-test"
    "performance-test"
    "security-test"
    "monitoring-test"
    "benchmark"
    "test"
)

# Clean up namespaces
echo "Cleaning up test namespaces..."
for ns in "${TEST_NAMESPACES[@]}"; do
    if kubectl get namespace "$ns" &>/dev/null; then
        echo -n "Deleting namespace $ns... "
        if kubectl delete namespace "$ns" --grace-period=0 --force &>/dev/null; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${YELLOW}⚠${NC} (may still be terminating)"
        fi
    fi
done

# Clean up test jobs in other namespaces
echo -e "\nCleaning up test jobs..."
for job in $(kubectl get jobs --all-namespaces -o json | jq -r '.items[] | select(.metadata.name | test("test|bench")) | "\(.metadata.namespace)/\(.metadata.name)"'); do
    ns=$(echo "$job" | cut -d'/' -f1)
    name=$(echo "$job" | cut -d'/' -f2)
    echo -n "Deleting job $name in namespace $ns... "
    if kubectl delete job "$name" -n "$ns" &>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
done

# Clean up test pods
echo -e "\nCleaning up test pods..."
for pod in $(kubectl get pods --all-namespaces -o json | jq -r '.items[] | select(.metadata.name | test("test|debug|bench")) | "\(.metadata.namespace)/\(.metadata.name)"'); do
    ns=$(echo "$pod" | cut -d'/' -f1)
    name=$(echo "$pod" | cut -d'/' -f2)
    if [[ ! "$ns" =~ ^(kube-system|kube-public|kube-node-lease)$ ]]; then
        echo -n "Deleting pod $name in namespace $ns... "
        if kubectl delete pod "$name" -n "$ns" --grace-period=0 --force &>/dev/null; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${YELLOW}⚠${NC}"
        fi
    fi
done

# Clean up test PVCs
echo -e "\nCleaning up test PVCs..."
for pvc in $(kubectl get pvc --all-namespaces -o json | jq -r '.items[] | select(.metadata.name | test("test|bench")) | "\(.metadata.namespace)/\(.metadata.name)"'); do
    ns=$(echo "$pvc" | cut -d'/' -f1)
    name=$(echo "$pvc" | cut -d'/' -f2)
    echo -n "Deleting PVC $name in namespace $ns... "
    if kubectl delete pvc "$name" -n "$ns" &>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
done

echo -e "\n✅ Cleanup complete!"