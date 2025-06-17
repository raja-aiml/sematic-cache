# End-to-End Testing Framework

Production-ready testing framework for the Semantic Cache Kubernetes deployment.

## 🎯 Overview

This modular testing framework provides comprehensive validation of:
- **Infrastructure**: Cluster, namespaces, databases, caching, ingress
- **Application**: Deployments, services, scaling, configuration
- **API**: Endpoints, functionality, error handling, concurrency
- **Performance**: Response times, load handling, resource usage
- **Monitoring**: Logging, metrics, health checks, observability

## 📁 Structure

```
deploy/tests/
├── e2e.sh                    # Main test orchestrator (50 lines)
├── lib/                      # Shared utilities
│   ├── config.sh            # Configuration and constants
│   ├── utils.sh             # Common utility functions
│   └── test-framework.sh    # Test framework and HTTP utilities
├── suites/                   # Test suite modules
│   ├── infrastructure.sh    # Infrastructure component tests
│   ├── application.sh       # Application deployment tests
│   ├── api.sh               # API endpoint and functionality tests
│   ├── performance.sh       # Performance and load tests
│   └── monitoring.sh        # Monitoring and observability tests
└── README.md                # This documentation
```

## 🚀 Quick Start

### Complete Workflow
```bash
# 1. Create cluster and infrastructure
deploy/cluster.sh up

# 2. Build and deploy application  
deploy/dev.sh build
deploy/dev.sh deploy

# 3. Run comprehensive tests
deploy/tests/e2e.sh all

# 4. Cleanup (optional)
deploy/cluster.sh down
```

### Selective Testing
```bash
# Test specific components
deploy/tests/e2e.sh infrastructure
deploy/tests/e2e.sh application api
deploy/tests/e2e.sh performance

# Quick essential tests only
deploy/tests/e2e.sh --quick all

# Custom timeout and verbose output
deploy/tests/e2e.sh --timeout 600 --verbose all
```

## 📋 Test Suites

### Infrastructure Tests
- **Cluster**: Creation, accessibility, node health
- **Namespaces**: Proper namespace creation and isolation
- **PostgreSQL**: Deployment, connectivity, pgvector extension
- **Redis**: Deployment, connectivity, ping response
- **Ingress Controller**: Deployment, service availability

### Application Tests  
- **Deployments**: Pod creation, readiness, scaling
- **Services**: Service creation, endpoint availability
- **Ingress**: Route configuration, external accessibility
- **ConfigMaps**: Web content and configuration
- **Secrets**: API keys and database credentials

### API Tests
- **Health Endpoints**: `/health`, `/metrics`, web interface
- **Cache Operations**: SET, GET, cache miss handling
- **Advanced Features**: Query, top-K, embedding search
- **Error Handling**: Invalid input, missing fields
- **Concurrency**: Parallel request handling

### Performance Tests
- **Response Time**: Average API response latency
- **Concurrent Load**: Multiple simultaneous requests
- **Resource Usage**: Memory and CPU consumption
- **Load Handling**: Graduated load testing
- **Cache Performance**: Cache retrieval speed

### Monitoring Tests
- **Logging**: Application and infrastructure log generation
- **Metrics**: Resource usage and availability
- **Health Checks**: Readiness and liveness probes
- **Events**: Kubernetes event monitoring
- **Network**: Inter-pod connectivity

## ⚙️ Configuration

### Environment Variables
```bash
# Override default settings
export BASE_URL="http://localhost:8080"
export TEST_TIMEOUT=600
export HEALTH_CHECK_RETRIES=50
```

### Test Configuration
Edit `lib/config.sh` to modify:
- Cluster and namespace names
- API endpoints and test data
- Timeout and retry settings
- Output colors and formatting

## 🔧 Customization

### Adding New Tests
1. **Create test function** in appropriate suite file:
```bash
test_my_feature() {
    log "🔧 Testing my feature..."
    
    # Your test logic here
    if my_test_condition; then
        success "My feature works correctly"
    else
        error "My feature failed"
    fi
}
```

2. **Export function** at bottom of suite file:
```bash
export -f test_my_feature
```

3. **Add to test suite** in `e2e.sh`:
```bash
run_test_suite "My Suite" \
    test_my_feature \
    test_other_features
```

### Creating New Test Suite
1. **Create new suite file**: `suites/mysuite.sh`
2. **Source framework**: `source "$SCRIPT_DIR/../lib/test-framework.sh"`
3. **Implement test functions** with proper exports
4. **Add suite to main orchestrator** in `e2e.sh`

## 📊 Test Results

### Success Indicators
- ✅ **Green checkmarks**: Tests passed
- **Response times**: Under acceptable thresholds
- **Resource usage**: Within configured limits
- **Zero failures**: All components healthy

### Warning Indicators  
- ⚠️ **Yellow warnings**: Non-critical issues
- **Slow responses**: Performance concerns
- **Missing features**: Optional components unavailable

### Failure Indicators
- ❌ **Red errors**: Critical test failures
- **Connection failures**: Infrastructure issues
- **API errors**: Application problems
- **Resource exhaustion**: Capacity issues

### Summary Report
```
==================================
📊 TEST SUMMARY
==================================
✅ Passed: 45
❌ Failed: 2
📊 Total:  47

❌ Failed tests:
   - PostgreSQL connection failed
   - Cache retrieval performance: 1200ms average

🚨 Some tests failed. Check the output above for details.
```

## 🐛 Troubleshooting

### Common Issues

**Cluster Not Found**
```bash
# Verify cluster exists
k3d cluster list

# Create if missing
deploy/cluster.sh up
```

**Connection Timeouts**
```bash
# Increase timeout
deploy/tests/e2e.sh --timeout 600 all

# Check cluster health
kubectl cluster-info
```

**Pod Not Ready**
```bash
# Check pod status
kubectl get pods -A

# View pod logs
kubectl logs -n app deployment/sematic-cache
```

**API Not Responding**
```bash
# Check service status
kubectl get svc -n app

# Test port forwarding
kubectl port-forward -n app svc/sematic-cache 8080:8080
```

### Debug Mode
```bash
# Enable verbose output
deploy/tests/e2e.sh --verbose api

# Check specific component
kubectl describe pod -n app -l app=sematic-cache
```

## 🏆 Best Practices

### Test Development
- **Small, focused tests**: Single responsibility per function
- **Clear naming**: Descriptive function and variable names
- **Proper error handling**: Meaningful error messages
- **Resource cleanup**: Reset state between tests

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Run E2E Tests
  run: |
    deploy/cluster.sh up
    deploy/dev.sh build
    deploy/dev.sh deploy
    deploy/tests/e2e.sh --quick all
    deploy/cluster.sh down
```

### Performance Baselines
- **Response time**: < 1000ms for API calls
- **Memory usage**: < 512Mi per pod
- **CPU usage**: < 500m per pod  
- **Success rate**: > 95% under load

## 📈 Metrics and Monitoring

### Key Performance Indicators
- **API availability**: 99.9% uptime target
- **Response time**: P95 < 500ms
- **Error rate**: < 1% of requests
- **Resource efficiency**: Optimal CPU/memory usage

### Observability
- **Structured logging**: JSON format with log levels
- **Metrics collection**: Prometheus-compatible endpoints
- **Health checks**: Kubernetes readiness/liveness probes
- **Distributed tracing**: Request correlation IDs

---

## 💡 Usage Examples

### Development Workflow
```bash
# Quick development cycle
deploy/cluster.sh up
deploy/dev.sh build && deploy/dev.sh deploy
deploy/tests/e2e.sh --quick api

# Make changes, then:
deploy/dev.sh build && deploy/dev.sh deploy
deploy/tests/e2e.sh api
```

### Pre-Production Validation
```bash
# Full comprehensive testing
deploy/tests/e2e.sh all

# Focus on critical paths
deploy/tests/e2e.sh infrastructure application api

# Performance validation
deploy/tests/e2e.sh performance monitoring
```

### Continuous Integration
```bash
# Automated pipeline testing
deploy/tests/e2e.sh --timeout 1200 --quick all
```

This testing framework ensures your Semantic Cache deployment is production-ready with comprehensive validation across all components! 🚀