# Production-Quality Local Development Setup Guide

This guide provides recommendations for establishing a production-quality local development environment for the Semantic Cache deployment tools.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Security Best Practices](#security-best-practices)
6. [Development Workflow](#development-workflow)
7. [Testing Strategy](#testing-strategy)
8. [Monitoring & Observability](#monitoring--observability)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# Clone and setup
git clone <repository>
cd deploy/local

# Install dependencies
make deps

# Run all checks
make check

# Build binary to bin/ directory
make build

# Run the development environment
make dev-up
```

## Architecture Overview

### Component Structure

```
deploy/local/
├── bin/                    # Build output (git-ignored)
├── cmd/                    # CLI commands
├── pkg/                    # Shared packages
│   ├── cmd/               # Command utilities
│   ├── config/            # Configuration management
│   ├── constants/         # Shared constants
│   ├── docker/            # Docker operations
│   ├── k3d/               # K3d cluster management
│   ├── kubernetes/        # K8s client operations
│   ├── secrets/           # Secret management
│   ├── testing/           # Test utilities
│   └── utils/             # Common utilities
├── config/                # Configuration files
├── logs/                  # Application logs (git-ignored)
└── releases/              # Release artifacts (git-ignored)
```

### Key Design Principles

1. **Separation of Concerns**: Each package has a single, well-defined responsibility
2. **Dependency Injection**: All external dependencies are injected via interfaces
3. **Error Handling**: Comprehensive error handling with context
4. **Configuration**: Flexible configuration with validation
5. **Security**: Defense-in-depth approach with multiple security layers

## Installation

### Prerequisites

- Go 1.21+
- Docker Desktop or Docker Engine
- kubectl
- k3d v5.x
- make
- golangci-lint (for development)

### Installation Steps

```bash
# Install Go dependencies
make deps

# Install development tools
go install github.com/golangci-lint/golangci-lint/cmd/golangci-lint@latest

# Build the binary
make build

# Optional: Install to PATH
make install
```

### Verification

```bash
# Check version
./bin/semantic-cache-deploy version

# Run health check
./bin/semantic-cache-deploy cluster test
```

## Configuration

### Configuration Files

1. **deploy-config.yaml**: Main configuration file
2. **.env**: Environment variables (secrets)
3. **Makefile**: Build and development tasks

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-api-key"

# Optional
export SC_DEPLOY_DEBUG=true
export SC_DEPLOY_CLUSTER_NAME="my-cluster"
export SC_DEPLOY_BUILD_TIMEOUT="15m"
```

### Configuration Validation

```bash
# Validate configuration
./bin/semantic-cache-deploy config validate

# Export effective configuration
./bin/semantic-cache-deploy config export
```

## Security Best Practices

### 1. Secret Management

```bash
# Use environment variables
export OPENAI_API_KEY="$(vault read -field=key secret/openai)"

# Or use a secret file
echo "OPENAI_API_KEY=xxx" > .env
chmod 600 .env
```

### 2. RBAC Configuration

```yaml
# Enable RBAC in config
security:
  enable_rbac: true
  enable_psp: true
```

### 3. Network Policies

```bash
# Apply network policies
kubectl apply -f deploy/k8s/network-policies/
```

### 4. Image Scanning

```bash
# Scan images before deployment
docker scan semantic-cache:local
```

### 5. Audit Logging

```yaml
# Enable audit logging
security:
  audit_logging: true
```

## Development Workflow

### Standard Development Cycle

```bash
# 1. Start development environment
make dev-up

# 2. Make changes to code
vim cmd/cluster.go

# 3. Run checks
make check

# 4. Build and test
make build test

# 5. Deploy changes
make dev

# 6. View logs
make dev-logs

# 7. Clean up
make dev-down
```

### Git Workflow

```bash
# Feature branch workflow
git checkout -b feature/my-feature

# Make changes and test
make check

# Commit with conventional commits
git commit -m "feat: add new functionality"

# Push and create PR
git push origin feature/my-feature
```

## Testing Strategy

### Unit Tests

```go
// Example test structure
func TestClusterManager_CreateCluster(t *testing.T) {
    tests := []struct {
        name    string
        setup   func()
        want    error
        wantErr bool
    }{
        {
            name: "successful creation",
            setup: func() {
                // Mock setup
            },
            wantErr: false,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // Test implementation
        })
    }
}
```

### Integration Tests

```bash
# Run integration tests
make test-integration

# Run with coverage
make test-coverage
```

### End-to-End Tests

```bash
# Run full e2e suite
./bin/semantic-cache-deploy workflow test

# Run specific test suite
./bin/semantic-cache-deploy test api
```

### Performance Tests

```bash
# Run benchmarks
make benchmark

# Profile CPU usage
go test -cpuprofile=cpu.prof -bench=.
go tool pprof cpu.prof
```

## Monitoring & Observability

### Structured Logging

```go
// Use structured logging
logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
    Level: slog.LevelDebug,
}))

logger.Info("cluster created",
    "cluster", clusterName,
    "duration", time.Since(start),
)
```

### Metrics Collection

```go
// Prometheus metrics
var (
    clusterCreateDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "cluster_create_duration_seconds",
            Help: "Time taken to create cluster",
        },
        []string{"cluster_name"},
    )
)
```

### Distributed Tracing

```go
// OpenTelemetry integration
tracer := otel.Tracer("semantic-cache-deploy")
ctx, span := tracer.Start(ctx, "CreateCluster")
defer span.End()
```

### Health Checks

```bash
# Check cluster health
./bin/semantic-cache-deploy cluster test

# Check application health
curl http://localhost:8080/health
```

## Performance Optimization

### 1. Parallel Operations

```go
// Execute operations in parallel
g, ctx := errgroup.WithContext(ctx)

g.Go(func() error {
    return createNamespace(ctx, "app")
})

g.Go(func() error {
    return createNamespace(ctx, "infra")
})

if err := g.Wait(); err != nil {
    return err
}
```

### 2. Connection Pooling

```go
// Reuse Kubernetes client
var clientInstance *kubernetes.Client
var clientOnce sync.Once

func GetClient() *kubernetes.Client {
    clientOnce.Do(func() {
        clientInstance, _ = kubernetes.NewClient("")
    })
    return clientInstance
}
```

### 3. Caching

```go
// Cache expensive operations
type Cache struct {
    mu    sync.RWMutex
    items map[string]CacheItem
}

func (c *Cache) Get(key string) (interface{}, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    item, ok := c.items[key]
    return item.Value, ok
}
```

### 4. Resource Limits

```yaml
# Set resource limits
cluster:
  resource_limits:
    cpu_limit: "2"
    memory_limit: "4Gi"
```

## Troubleshooting

### Common Issues

#### 1. Cluster Creation Fails

```bash
# Check k3d logs
k3d cluster list
docker logs k3d-semantic-cache-server-0

# Reset and retry
make dev-down
make clean
make dev-up
```

#### 2. Build Failures

```bash
# Clean and rebuild
make clean deps build

# Check for dependency issues
go mod tidy
go mod verify
```

#### 3. Deployment Issues

```bash
# Debug deployment
./bin/semantic-cache-deploy debug analyze full

# Check pod logs
kubectl logs -n app -l app=semantic-cache --tail=100
```

#### 4. Performance Issues

```bash
# Profile the application
go tool pprof http://localhost:6060/debug/pprof/profile

# Check resource usage
kubectl top nodes
kubectl top pods -n app
```

### Debug Mode

```bash
# Enable debug logging
export DEBUG=1
export SC_DEPLOY_DEBUG=true

# Run with verbose output
./bin/semantic-cache-deploy -v cluster up
```

### Support Resources

- **Documentation**: See README.md and inline help
- **Issues**: GitHub Issues for bug reports
- **Logs**: Check `./logs/` directory
- **Community**: Join our Slack channel

## Production Readiness Checklist

Before deploying to production:

- [ ] All tests passing (`make check`)
- [ ] Security scan completed
- [ ] Configuration validated
- [ ] Resource limits set
- [ ] Monitoring configured
- [ ] Backup strategy defined
- [ ] Rollback plan documented
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Change log updated

## Continuous Improvement

### Regular Maintenance

1. **Weekly**: Update dependencies
2. **Monthly**: Security audit
3. **Quarterly**: Performance review
4. **Annually**: Architecture review

### Metrics to Track

- Deployment success rate
- Mean time to recovery (MTTR)
- Resource utilization
- Error rates
- Performance benchmarks

### Feedback Loop

1. Collect metrics
2. Analyze trends
3. Identify improvements
4. Implement changes
5. Measure impact

---

This production-quality setup provides a robust foundation for local Kubernetes development with emphasis on reliability, security, and developer experience.