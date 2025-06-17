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

### Complete Workflow with Task
```bash
# 1. Create cluster and infrastructure
task setup

# 2. Build and deploy application  
task build deploy

# 3. Run comprehensive tests
task test

# 4. Cleanup (optional)
task cleanup
```

### One-Command Workflows
```bash
# Complete production workflow
task full

# Quick development cycle
task quick

# Production readiness validation
task production-ready
```

### Selective Testing with Task
```bash
# Test specific components
task test:infrastructure
task test:application
task test:api
task test:performance

# Quick essential tests only
task test:quick

# Custom testing combinations
task test:infrastructure test:api
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

## ⚙️ Task Integration

### Core Testing Tasks
```bash
# Individual test suites
task test:infrastructure    # Test cluster and infrastructure
task test:application      # Test deployments and services
task test:api             # Test API endpoints
task test:performance     # Test performance and load
task test:monitoring      # Test logging and observability

# Combined workflows
task test                 # Run all comprehensive tests
task test:quick          # Run essential tests only
task verify              # Quick verification of deployment
```

### Development Tasks
```bash
# Development cycle
task dev                 # build → deploy → quick-test
task redeploy           # build → deploy
task quick-start        # setup → build → deploy

# Monitoring and debugging
task status             # Deployment status
task logs              # Application logs
task health            # Quick health check
task debug             # Comprehensive debug info
task debug:pods        # Pod-specific debugging
task debug:network     # Network connectivity debugging
```

### Advanced Operations
```bash
# Scaling operations
task scale:up          # Scale to 2 replicas
task scale:down        # Scale to 1 replica
task restart           # Restart deployment

# Validation and maintenance
task validate          # Validate configurations
task ci:test          # CI-friendly testing
task backup:config    # Backup current config
task update:images    # Update application images
```

## 🔧 Configuration

### Environment Variables
```bash
# Override default settings
export BASE_URL="http://localhost:8080"
export TEST_TIMEOUT=600
export HEALTH_CHECK_RETRIES=50
```

### Task Variables
Edit `Taskfile.yaml` to modify:
```yaml
vars:
  CLUSTER_NAME: sematic-cache
  API_URL: http://localhost:8080/semantic-cache
  WEB_URL: http://localhost:8080/web
```

### Test Configuration
Edit `lib/config.sh` to modify:
- Cluster and namespace names
- API endpoints and test data
- Timeout and retry settings
- Output colors and formatting

## 🎯 Usage Patterns

### Development Workflow
```bash
# Quick development cycle
task quick-start          # Initial setup
task dev                 # Iterative development
task test:api           # Test changes
task logs               # Debug issues
```

### Pre-Production Validation
```bash
# Full comprehensive testing
task production-ready

# Focus on critical paths
task test:infrastructure test:application test:api

# Performance validation
task test:performance task:monitoring
```

### Continuous Integration
```bash
# Automated pipeline testing
task ci:test
task validate
```

### Debugging and Troubleshooting
```bash
# General debugging
task debug
task status
task logs

# Specific issue debugging
task debug:pods         # Pod issues
task debug:network     # Network issues
task verify            # Overall health
```

## 📊 Task Dependencies

### Workflow Dependencies
```yaml
# Automatic dependency resolution
quick-start:
  deps: [setup, build, deploy]

full-cycle:
  deps: [build, deploy, test:quick]

production-ready:
  deps: [setup, build, deploy, test, verify]
```

### Parallel Execution
```bash
# Tasks can run in parallel when safe
task test:infrastructure test:performance  # Parallel execution
task build deploy                         # Sequential (deploy depends on build)
```

## 🐛 Troubleshooting with Task

### Common Issues and Solutions

**Cluster Not Found**
```bash
task cluster:info       # Check cluster status
task setup             # Create if missing
```

**Connection Timeouts**
```bash
task verify            # Quick verification
task debug:network     # Network diagnostics
```

**Pod Not Ready**
```bash
task debug:pods        # Pod-specific debug
task logs              # Check application logs
task restart           # Restart if needed
```

**API Not Responding**
```bash
task health            # Quick health check
task port-forward:api  # Direct port forwarding
task debug             # Comprehensive debug
```

### Debug Tasks
```bash
# Progressive debugging
task verify            # Quick overall check
task status           # Detailed status
task debug            # Comprehensive debug info
task debug:pods       # Pod-specific details
task debug:network    # Network connectivity
```

## 🏆 Best Practices with Task

### Task Development
- **Descriptive names**: Use clear, hierarchical task names
- **Proper dependencies**: Define task dependencies correctly
- **Error handling**: Tasks fail fast with meaningful messages
- **Documentation**: Each task has a clear description

### Workflow Optimization
```bash
# Use task dependencies for complex workflows
task production-ready  # Runs: setup → build → deploy → test → verify

# Combine tasks for efficiency
task test:infrastructure test:api  # Run multiple suites

# Use variables for consistency
{{.API_URL}}/health    # Consistent URL usage
```

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Run E2E Tests
  run: |
    task setup
    task build deploy
    task ci:test
    task cleanup
```

## 📈 Performance Baselines

### Key Performance Indicators
- **API availability**: 99.9% uptime target
- **Response time**: P95 < 500ms
- **Error rate**: < 1% of requests
- **Resource efficiency**: Optimal CPU/memory usage

### Task-based Monitoring
```bash
# Regular monitoring tasks
task health            # Quick health check
task metrics          # Performance metrics
task resources        # Resource usage
task api:benchmark    # Performance benchmark
```

## 💡 Advanced Task Usage

### Custom Task Combinations
```bash
# Create custom workflows
task setup build deploy test:api verify

# Environment-specific testing
task test:infrastructure --verbose
task test:performance --timeout 1200
```

### Task Aliases and Shortcuts
```bash
# Common shortcuts defined in Taskfile.yaml
task clean            # Alias for cleanup
task dev              # Development cycle
task ci:test         # CI-friendly testing
```

### Conditional Tasks
Tasks can include conditions and error handling:
```yaml
verify:
  cmds:
    - k3d cluster list | grep {{.CLUSTER_NAME}} && echo "✅ Cluster exists" || echo "❌ Cluster missing"
    - curl -s --max-time 5 {{.API_URL}}/health >/dev/null && echo "✅ API responding" || echo "❌ API not responding"
```

This Task-based testing framework provides a modern, efficient way to manage your deployment workflow with clear dependencies, parallel execution, and comprehensive testing coverage! 🚀