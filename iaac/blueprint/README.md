# Blueprint

**Professional Kubernetes blueprint with Istio service mesh and comprehensive observability**

A production-ready Kubernetes blueprint built on k3d that provides a complete development and testing environment with PostgreSQL, Redis, Istio service mesh, and comprehensive observability stack.

## 🚀 Quick Start

```bash
# Clone and navigate to blueprint
cd iaac/blueprint

# Create cluster and deploy minimal stack
task setup
task deploy:minimal

# Check status
task status

# Run smoke tests
task test:smoke
```

## 📋 Features

### Core Infrastructure
- **PostgreSQL** with pgvector extension for semantic search
- **Redis** for high-performance caching
- **Istio Service Mesh** for advanced traffic management
- **Comprehensive Observability** with Prometheus, Grafana, Loki, and OpenTelemetry

### Deployment Scenarios
- **Minimal**: Just the essentials (PostgreSQL + Redis)
- **Development**: Full dev experience with debugging tools
- **Service Mesh**: Istio-focused deployment with advanced networking
- **Monitoring Only**: Pure observability stack
- **Full Stack**: Everything enabled for production-like testing

### DevOps Features
- **Kustomize-based** configuration management
- **Environment-specific** overlays
- **Comprehensive testing** and validation
- **Automated scripts** for common operations
- **Resource governance** with quotas and limits

## 📁 Project Structure

```
blueprint/
├── infra/                      # 🏗️ Kustomize-based Kubernetes manifests
│   ├── base/                   # Environment-agnostic base stack
│   ├── overlays/               # Environment-specific configurations
│   └── modules/                # Optional add-on components
├── scenarios/                  # 🌐 Real-world blueprint compositions
├── scripts/                    # ⚙️ Automation and ops tools
├── validation-kit/             # ✅ Testing and validation
└── hack/                       # 🔧 Dev and maintenance scripts
```

## 🎯 Scenarios

### Minimal Scenario
Perfect for local development and testing:
```bash
task scenarios:minimal
```
- PostgreSQL with pgvector
- Redis for caching
- Basic networking and security

### Development Scenario
Full development experience:
```bash
task scenarios:development
```
- All minimal components
- Debug tools and utilities
- Enhanced logging and monitoring
- Development-friendly configurations

### Service Mesh Scenario
Istio-focused deployment:
```bash
task scenarios:service-mesh
```
- Complete Istio service mesh
- Advanced traffic management
- Security policies and mTLS
- Observability integration

### Full Stack Scenario
Production-like environment:
```bash
task scenarios:full-stack
```
- All components enabled
- Complete observability stack
- Security hardening
- Performance monitoring

## 🛠️ Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [k3d](https://k3d.io/v5.4.4/#installation)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Task](https://taskfile.dev/installation/) (optional, but recommended)

## 📖 Common Tasks

### Cluster Management
```bash
# Create cluster with registry
task setup

# Show cluster information
task cluster:info

# Destroy cluster
task cleanup
```

### Deployments
```bash
# Deploy different scenarios
task scenarios:minimal
task scenarios:development
task scenarios:service-mesh
task scenarios:full-stack

# Check deployment status
task status
task health
```

### Testing & Validation
```bash
# Run smoke tests
task test:smoke

# Run comprehensive tests
task test

# Run integration tests
task test:integration
```

### Development
```bash
# Development workflow
task dev

# Port forward to services
task port-forward:postgres   # localhost:5432
task port-forward:redis      # localhost:6379

# View logs
task logs
```

## 🔧 Configuration

### Environment Variables
Configuration files are now centralized in the `iaac/config` directory:
```bash
# Copy and customize the environment file
cp ../config/blueprint.env.example ../config/blueprint.env

# Or use the iaac CLI with config options
iaac --config-dir ../config cluster up
```

Key configurations:
- `CLUSTER_NAME`: Name of the k3d cluster
- `POSTGRES_PASSWORD`: Database password
- `REDIS_PASSWORD`: Redis password
- `GRAFANA_ADMIN_PASSWORD`: Grafana admin password

### Customization
Use Kustomize overlays to customize deployments:
```bash
# Local development overlay
kubectl apply -k infra/overlays/local/

# Development overlay with full features
kubectl apply -k infra/overlays/dev/
```

## 🔍 Monitoring & Observability

### Access Dashboards
```bash
# Port forward to Grafana (when deployed)
kubectl port-forward svc/grafana 3000:3000 -n monitoring

# Port forward to Prometheus (when deployed)
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
```

### Available Dashboards
- Istio Service Mesh Overview
- Kubernetes Cluster Metrics
- PostgreSQL Performance
- Redis Monitoring
- Application Traces (OpenTelemetry)

## 🧪 Testing

### Smoke Tests
Quick validation of essential components:
```bash
./validation-kit/scripts/smoke-test.sh
```

### Integration Tests
Comprehensive testing across all components:
```bash
./validation-kit/scripts/integration-test.sh
```

### Manual Testing
```bash
# Test PostgreSQL connectivity
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
  psql -h postgres.infra.svc.cluster.local -U cache -d cache

# Test Redis connectivity
kubectl run -it --rm debug --image=redis:7 --restart=Never -- \
  redis-cli -h redis.infra.svc.cluster.local
```

## 🔒 Security

### Network Policies
Default deny-all policies with explicit allow rules for necessary communication.

### Pod Security
Security contexts and pod security standards applied to all workloads.

### Secrets Management
Sensitive data stored in Kubernetes secrets with proper RBAC controls.

## 🚨 Troubleshooting

### Common Issues

**Cluster won't start:**
```bash
# Check Docker is running
docker ps

# Recreate cluster
task cleanup
task setup
```

**Pods stuck in pending:**
```bash
# Check node resources
kubectl describe nodes

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp
```

**Network issues:**
```bash
# Debug network policies
task debug:network

# Check service connectivity
task validation-kit/client-connections/
```

### Debug Tools
```bash
# Deploy debug pod
kubectl apply -f validation-kit/client-connections/debug-pod.yaml

# Access debug pod
kubectl exec -it debug-pod -- /bin/bash
```

## 📚 Documentation

- [Validation Kit](validation-kit/README.md) - Testing and validation
- [Scripts Documentation](scripts/README.md) - Automation scripts
- [Scenarios Guide](scenarios/) - Detailed scenario documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test your changes with all scenarios
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [k3d](https://k3d.io/) - k3s in Docker
- [Istio](https://istio.io/) - Service mesh
- [Kustomize](https://kustomize.io/) - Configuration management
- [PostgreSQL](https://www.postgresql.org/) - Database
- [Redis](https://redis.io/) - In-memory cache