# K3D Blueprint Validation Kit

Comprehensive testing and validation suite for the K3D Blueprint.

## Overview

The Validation Kit provides:
- Integration tests for all components
- Performance benchmarks
- Security validation
- Observability verification
- Client connection tests
- Seed data for testing

## Structure

```
validation-kit/
├── tests/              # Test definitions
│   ├── integration/    # Component integration tests
│   ├── performance/    # Load and stress tests
│   ├── security/       # Security validation
│   └── observability/  # Metrics, logs, traces tests
├── client-connections/ # Test pods for service verification
├── seed-data/         # Sample data for testing
└── scripts/           # Test automation scripts
```

## Quick Start

### Run All Tests
```bash
./scripts/run-tests.sh
```

### Run Specific Test Suite
```bash
# Smoke tests only
./scripts/smoke-test.sh

# Integration tests
./scripts/integration-test.sh

# Performance tests
./scripts/performance-test.sh

# Security tests
./scripts/security-test.sh
```

## Test Categories

### 1. Integration Tests
Verify component connectivity and basic functionality:
- PostgreSQL connectivity and operations
- Redis connectivity and operations
- Service discovery
- Network policies
- Ingress routing

### 2. Performance Tests
Benchmark and stress test components:
- Database throughput
- Cache performance
- Network latency
- Resource utilization
- Scalability limits

### 3. Security Tests
Validate security configurations:
- mTLS enforcement
- Network isolation
- RBAC policies
- Pod security standards
- Secret management

### 4. Observability Tests
Ensure monitoring and logging work correctly:
- Metrics collection
- Log aggregation
- Distributed tracing
- Alert firing
- Dashboard functionality

## Client Connections

Pre-configured test pods for manual verification:

```bash
# PostgreSQL client
kubectl apply -f client-connections/postgres-client.yaml
kubectl exec -it postgres-client -- psql -h postgres.infra -U postgres

# Redis client
kubectl apply -f client-connections/redis-client.yaml
kubectl exec -it redis-client -- redis-cli -h redis.infra

# General debugging
kubectl apply -f client-connections/debug-pod.yaml
kubectl exec -it debug-pod -- bash
```

## Seed Data

Sample data for testing various scenarios:

### PostgreSQL
- Schema definitions
- Sample data sets
- Performance test data
- Query examples

### Redis
- Key-value pairs
- Data structures
- Cache patterns
- Benchmark data

### Monitoring
- Metric examples
- Alert configurations
- Dashboard queries
- Log samples

## Writing New Tests

### Test Structure
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: test-example
  namespace: test
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: test
        image: appropriate/image
        command: ["test-command"]
        env:
        - name: TEST_VAR
          value: "test-value"
```

### Best Practices
1. Use Jobs for test execution
2. Include proper cleanup
3. Set appropriate timeouts
4. Use meaningful assertions
5. Generate clear reports

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run K3D Blueprint Tests
  run: |
    ./validation-kit/scripts/run-tests.sh
  timeout-minutes: 30
```

### Jenkins Pipeline Example
```groovy
stage('Validation') {
    steps {
        sh './validation-kit/scripts/run-tests.sh'
    }
    post {
        always {
            junit 'test-results/*.xml'
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Tests timeout**: Increase wait times or check resource constraints
2. **Connection refused**: Verify services are running and network policies
3. **Permission denied**: Check RBAC and service accounts
4. **Resource exhausted**: Scale down test parallelism

### Debug Commands
```bash
# Check test pod logs
kubectl logs -n test <pod-name>

# Watch test progress
kubectl get pods -n test -w

# Clean up failed tests
./scripts/cleanup.sh
```

## Contributing

To add new tests:
1. Create test definition in appropriate category
2. Add to relevant test script
3. Update documentation
4. Test in isolation first
5. Submit PR with results