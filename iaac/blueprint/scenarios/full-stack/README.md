# Full-Stack Scenario

Complete production-ready deployment with all features enabled.

## What's Included

### Infrastructure
- PostgreSQL (HA with streaming replication)
- Redis (Cluster mode with 3 nodes)
- Persistent storage with backups

### Service Mesh
- Istio with production profile
- mTLS enforcement
- Advanced traffic management
- Circuit breaking and retry policies

### Observability
- Prometheus with long-term retention
- Grafana with full dashboard suite
- Loki for centralized logging
- Distributed tracing with Tempo
- Alerting with PagerDuty integration

### Security
- Pod security policies
- Network segmentation
- RBAC with least privilege
- Secrets encryption at rest
- Admission controllers

## Deployment

```bash
# Ensure cluster has sufficient resources
# Recommended: 8 CPU, 16GB RAM minimum

# Deploy full stack
kubectl apply -k scenarios/full-stack

# Wait for all components to be ready
kubectl wait --for=condition=ready pod --all -n production --timeout=600s
kubectl wait --for=condition=ready pod --all -n istio-system --timeout=600s
kubectl wait --for=condition=ready pod --all -n monitoring --timeout=600s

# Verify deployment
kubectl get pods -A | grep -E "(production|istio-system|monitoring|logging|tracing)"
```

## Production Considerations

### High Availability
- Multiple replicas for all critical components
- Pod disruption budgets configured
- Anti-affinity rules to spread pods across nodes

### Performance
- Resource limits tuned for production workloads
- Horizontal pod autoscaling enabled
- Optimized JVM settings for Java components

### Security
- Regular security scanning
- Automated certificate rotation
- Audit logging enabled
- Compliance controls

### Backup & Recovery
- Automated PostgreSQL backups every 6 hours
- Redis persistence with AOF
- Prometheus data backed up to object storage
- Disaster recovery procedures documented

## Monitoring

### Key Metrics
- Service latency < 100ms p99
- Error rate < 0.1%
- CPU utilization < 70%
- Memory utilization < 80%

### Alerts
Critical alerts configured for:
- Service downtime
- High error rates
- Resource exhaustion
- Security violations

## Maintenance

### Upgrades
```bash
# Canary deployment for applications
kubectl set image deployment/app app=myapp:v2 -n production --record

# Rolling update for infrastructure
kubectl rollout restart deployment/postgres -n infra
kubectl rollout status deployment/postgres -n infra
```

### Scaling
```bash
# Manual scaling
kubectl scale deployment redis --replicas=5 -n infra

# Autoscaling
kubectl autoscale deployment app --min=3 --max=10 --cpu-percent=70 -n production
```