# Production Guide

This guide covers production deployment, security best practices, performance optimization, and monitoring for the Semantic Cache Infrastructure.

## Production Architecture

### Design Principles

1. **Single Responsibility**: Each package has one well-defined purpose
2. **Dependency Injection**: All external dependencies injected via interfaces
3. **Error Handling**: Comprehensive error wrapping with context
4. **Configuration Management**: Validation-enabled, environment-aware
5. **Security First**: Defense-in-depth approach with multiple layers

### Component Structure

```
├── cmd/                    # Entry points
├── pkg/
│   ├── k3d/               # Cluster management
│   ├── kubernetes/        # K8s operations
│   ├── docker/           # Container management
│   ├── secrets/          # Secret management
│   └── utils/            # Shared utilities
└── bin/                   # Compiled binaries (git-ignored)
```

## Security Best Practices

### Secret Management

```go
// Use environment variables
apiKey := os.Getenv("OPENAI_API_KEY")

// File-based secrets with proper permissions
secretFile := "/etc/secrets/api-key"
os.Chmod(secretFile, 0600)

// Integration with external vaults
vault := secrets.NewVaultClient()
secret, err := vault.Get("api-key")
```

### Container Security

- **Non-root execution**: Containers run as non-root users
- **Image scanning**: Scan images before deployment
- **Security policies**: PSP/Pod Security Standards enforced
- **Network policies**: Restrict pod-to-pod communication

### RBAC Configuration

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: semantic-cache-role
rules:
  - apiGroups: [""]
    resources: ["pods", "services"]
    verbs: ["get", "list", "create", "update"]
```

## Performance Optimization

### Resource Requirements

- **Minimum**: 2 CPU cores, 4GB RAM
- **Recommended**: 4 CPU cores, 8GB RAM
- **Storage**: 20GB available space

### Optimization Techniques

#### 1. Connection Pooling

```go
var (
    clientOnce sync.Once
    k8sClient  kubernetes.Interface
)

func GetClient() kubernetes.Interface {
    clientOnce.Do(func() {
        k8sClient = createClient()
    })
    return k8sClient
}
```

#### 2. Parallel Operations

```go
g, ctx := errgroup.WithContext(context.Background())

g.Go(func() error {
    return deployRedis(ctx)
})

g.Go(func() error {
    return deployPostgres(ctx)
})

if err := g.Wait(); err != nil {
    return fmt.Errorf("parallel deployment failed: %w", err)
}
```

#### 3. Caching Layer

```go
type Cache struct {
    mu    sync.RWMutex
    items map[string]CacheItem
}

func (c *Cache) Get(key string) (interface{}, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    item, found := c.items[key]
    return item.Value, found
}
```

## Monitoring and Observability

### Structured Logging

```go
logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
    Level: slog.LevelInfo,
}))

logger.Info("deployment started",
    "cluster", clusterName,
    "namespace", namespace,
    "version", version,
)
```

### Metrics Collection

```go
var (
    deploymentDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "deployment_duration_seconds",
            Help: "Duration of deployments",
        },
        []string{"component", "status"},
    )
)

// Record metrics
start := time.Now()
err := deploy()
status := "success"
if err != nil {
    status = "failure"
}
deploymentDuration.WithLabelValues("app", status).Observe(time.Since(start).Seconds())
```

### Health Checks

```go
func healthCheck() error {
    checks := []struct {
        name string
        fn   func() error
    }{
        {"kubernetes", checkKubernetes},
        {"redis", checkRedis},
        {"postgres", checkPostgres},
    }
    
    for _, check := range checks {
        if err := check.fn(); err != nil {
            return fmt.Errorf("%s health check failed: %w", check.name, err)
        }
    }
    return nil
}
```

## Production Deployment Checklist

### Pre-deployment

- [ ] Run all tests: `task test`
- [ ] Security scan: `task security-scan`
- [ ] Validate configuration
- [ ] Set resource limits
- [ ] Configure monitoring
- [ ] Document rollback plan

### Deployment

- [ ] Use versioned images
- [ ] Enable audit logging
- [ ] Configure backups
- [ ] Set up alerts
- [ ] Test health endpoints
- [ ] Verify metrics collection

### Post-deployment

- [ ] Monitor error rates
- [ ] Check resource usage
- [ ] Validate performance
- [ ] Test disaster recovery
- [ ] Update documentation

## Performance Benchmarks

| Operation | Target | Current |
|-----------|--------|---------|
| Cluster Creation | < 60s | 45-60s |
| App Deployment | < 45s | 30-45s |
| Full Workflow | < 3m | 2-3m |
| API Response | < 100ms | 50-80ms |

## Troubleshooting Production Issues

### High Memory Usage

```bash
# Check memory usage
kubectl top pods -n app

# Analyze memory profile
go tool pprof http://localhost:6060/debug/pprof/heap
```

### Slow Response Times

```bash
# Check latency metrics
curl http://localhost:9090/metrics | grep duration

# Enable trace logging
export SC_DEPLOY_DEBUG=true
```

### Deployment Failures

```bash
# Analyze deployment
iaac debug analyze full

# Check events
kubectl get events -n app --sort-by='.lastTimestamp'
```

## Disaster Recovery

### Backup Strategy

```bash
# Backup configuration
kubectl get all -n app -o yaml > backup.yaml

# Backup persistent data
pg_dump $DATABASE_URL > postgres-backup.sql
```

### Recovery Procedures

1. **Service Failure**: Automatic restart via Kubernetes
2. **Node Failure**: Pod rescheduling to healthy nodes
3. **Data Loss**: Restore from latest backup
4. **Complete Failure**: Rebuild from infrastructure code

## Security Hardening

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: semantic-cache-network-policy
spec:
  podSelector:
    matchLabels:
      app: semantic-cache
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: frontend
    ports:
    - protocol: TCP
      port: 8080
```

### Pod Security

```yaml
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
  containers:
  - name: app
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
```

## Maintenance Schedule

| Task | Frequency | Description |
|------|-----------|-------------|
| Dependency Updates | Weekly | Update Go modules and base images |
| Security Audit | Monthly | Scan for vulnerabilities |
| Performance Review | Quarterly | Analyze metrics and optimize |
| Architecture Review | Annually | Evaluate design decisions |

## Recommended Next Steps

### High Priority
1. **Implement comprehensive unit tests** (>80% coverage)
2. **Add distributed tracing** with OpenTelemetry
3. **Implement retry logic** with exponential backoff

### Medium Priority
4. **Enhance caching** for expensive operations
5. **Add circuit breakers** for external services
6. **Implement graceful shutdown** handling

### Low Priority
7. **Create Helm charts** for easier deployment
8. **Add multi-cluster** support
9. **Implement A/B testing** capabilities