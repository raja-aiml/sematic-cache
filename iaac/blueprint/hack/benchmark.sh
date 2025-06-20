#!/bin/bash
set -euo pipefail

# Benchmark Script
# Performance benchmarking for K3D Blueprint components

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "⚡ K3D Blueprint Performance Benchmark"
echo "===================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check prerequisites
if ! kubectl get nodes > /dev/null 2>&1; then
    echo -e "${RED}❌ Kubernetes cluster not accessible${NC}"
    exit 1
fi

# Configuration
NAMESPACE="benchmark"
DURATION="60"  # seconds
CONNECTIONS="10"
THREADS="2"

# Create benchmark namespace
echo "Setting up benchmark environment..."
kubectl create namespace $NAMESPACE 2>/dev/null || true

# Deploy test databases
echo -e "\n🚀 Deploying test instances..."
kubectl apply -k "$PROJECT_ROOT/scenarios/minimal" -n $NAMESPACE 2>/dev/null || true

# Wait for deployments
echo "Waiting for deployments..."
kubectl wait --for=condition=available --timeout=300s deployment/postgres -n $NAMESPACE 2>/dev/null || echo -e "${YELLOW}⚠${NC} PostgreSQL not ready"
kubectl wait --for=condition=available --timeout=300s deployment/redis -n $NAMESPACE 2>/dev/null || echo -e "${YELLOW}⚠${NC} Redis not ready"

# PostgreSQL Benchmark
echo -e "\n📊 PostgreSQL Benchmark"
echo "====================="

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: pgbench-init
  namespace: $NAMESPACE
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: pgbench
        image: postgres:16-alpine
        command:
        - pgbench
        - -i
        - -s
        - "50"
        - -h
        - postgres.$NAMESPACE
        - -U
        - postgres
        - postgres
        env:
        - name: PGPASSWORD
          value: postgres
EOF

echo "Initializing pgbench database..."
kubectl wait --for=condition=complete job/pgbench-init -n $NAMESPACE --timeout=300s

# Run PostgreSQL benchmark
echo -e "\nRunning PostgreSQL benchmark..."
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: pgbench-run
  namespace: $NAMESPACE
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: pgbench
        image: postgres:16-alpine
        command:
        - pgbench
        - -c
        - "$CONNECTIONS"
        - -j
        - "$THREADS"
        - -T
        - "$DURATION"
        - -P
        - "10"
        - -h
        - postgres.$NAMESPACE
        - -U
        - postgres
        - postgres
        env:
        - name: PGPASSWORD
          value: postgres
EOF

# Get PostgreSQL results
echo -e "\nPostgreSQL Results:"
kubectl wait --for=condition=complete job/pgbench-run -n $NAMESPACE --timeout=600s
kubectl logs job/pgbench-run -n $NAMESPACE | tail -20

# Redis Benchmark
echo -e "\n📊 Redis Benchmark"
echo "================="

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: redis-bench
  namespace: $NAMESPACE
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: redis-bench
        image: redis:7-alpine
        command:
        - redis-benchmark
        - -h
        - redis.$NAMESPACE
        - -p
        - "6379"
        - -c
        - "$CONNECTIONS"
        - -n
        - "100000"
        - -d
        - "100"
        - -t
        - set,get,incr,lpush,rpush,lpop,rpop,sadd,hset,zadd,zpopmin,lrange
        - -q
        - --csv
EOF

echo -e "\nRunning Redis benchmark..."
kubectl wait --for=condition=complete job/redis-bench -n $NAMESPACE --timeout=300s

echo -e "\nRedis Results (requests per second):"
echo "Operation,Requests/sec"
kubectl logs job/redis-bench -n $NAMESPACE

# Network Performance
echo -e "\n📊 Network Performance Test"
echo "=========================="

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: iperf-server
  namespace: $NAMESPACE
spec:
  containers:
  - name: iperf
    image: networkstatic/iperf3
    command: ["iperf3", "-s"]
---
apiVersion: v1
kind: Service
metadata:
  name: iperf-server
  namespace: $NAMESPACE
spec:
  selector:
    app: iperf-server
  ports:
  - port: 5201
EOF

sleep 5

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: iperf-client
  namespace: $NAMESPACE
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: iperf
        image: networkstatic/iperf3
        command: ["iperf3", "-c", "iperf-server.$NAMESPACE", "-t", "10", "-P", "4"]
EOF

echo -e "\nRunning network benchmark..."
kubectl wait --for=condition=complete job/iperf-client -n $NAMESPACE --timeout=120s 2>/dev/null || true
echo -e "\nNetwork Performance:"
kubectl logs job/iperf-client -n $NAMESPACE 2>/dev/null | grep -E "(sender|receiver)" || echo "Network test skipped"

# Resource Usage
echo -e "\n📊 Resource Usage During Tests"
echo "=============================="

echo -e "\nPostgreSQL Resource Usage:"
kubectl top pod -n $NAMESPACE -l app=postgres --no-headers 2>/dev/null || echo "Metrics not available"

echo -e "\nRedis Resource Usage:"
kubectl top pod -n $NAMESPACE -l app=redis --no-headers 2>/dev/null || echo "Metrics not available"

# Latency Tests
echo -e "\n📊 Latency Tests"
echo "==============="

# Create latency test pod
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: latency-test
  namespace: $NAMESPACE
spec:
  containers:
  - name: tools
    image: nicolaka/netshoot
    command: ["/bin/bash", "-c", "sleep 3600"]
EOF

kubectl wait --for=condition=ready pod/latency-test -n $NAMESPACE --timeout=60s

echo -e "\nPostgreSQL latency (10 pings):"
kubectl exec -n $NAMESPACE latency-test -- ping -c 10 postgres.$NAMESPACE | tail -1

echo -e "\nRedis latency (10 pings):"
kubectl exec -n $NAMESPACE latency-test -- ping -c 10 redis.$NAMESPACE | tail -1

# Summary Report
echo -e "\n📋 Benchmark Summary"
echo "==================="

echo "Configuration:"
echo "- Duration: ${DURATION}s"
echo "- Connections: $CONNECTIONS"
echo "- Threads: $THREADS"

# Cleanup
echo -e "\n🧹 Cleaning up..."
kubectl delete namespace $NAMESPACE --force --grace-period=0 2>/dev/null || true

echo -e "\n✅ Benchmark complete!"