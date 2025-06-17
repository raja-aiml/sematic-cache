#!/usr/bin/env bash
# =====================================
# 📋 TEST CONFIGURATION
# =====================================

# Cluster configuration
CLUSTER_NAME="sematic-cache"
KUBE_CONTEXT="k3d-${CLUSTER_NAME}"
NAMESPACE_INFRA="infra"
NAMESPACE_APP="app"

# Test configuration
BASE_URL="http://localhost:8080"
TEST_TIMEOUT=300
HEALTH_CHECK_RETRIES=30
HEALTH_CHECK_INTERVAL=10

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test data for API tests
TEST_PROMPT="What is Kubernetes?"
TEST_ANSWER="Kubernetes is a container orchestration platform for managing containerized applications at scale."
TEST_MODEL="test-model"