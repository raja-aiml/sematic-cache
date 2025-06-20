# Minimal Scenario

The minimal scenario provides just the essential components needed for basic semantic cache functionality:

- PostgreSQL with pgvector extension
- Redis for caching
- Basic network policies
- Resource quotas

## Components

### Infrastructure
- **PostgreSQL**: Primary database with vector search capabilities
- **Redis**: Fast in-memory cache for frequently accessed data

### Resource Usage
- CPU: ~200m
- Memory: ~384Mi
- Storage: ~1.5Gi

## Deployment

```bash
# Deploy minimal scenario
task scenarios:minimal

# Check status
task status

# Port forward to services
task port-forward:postgres  # localhost:5432
task port-forward:redis     # localhost:6379
```

## Use Cases

Perfect for:
- Local development
- Testing basic functionality
- Resource-constrained environments
- Getting started quickly

## Testing

```bash
# Run smoke tests
task test:smoke

# Manual testing
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- psql -h postgres.infra.svc.cluster.local -U cache -d cache
kubectl run -it --rm debug --image=redis:7 --restart=Never -- redis-cli -h redis.infra.svc.cluster.local
```