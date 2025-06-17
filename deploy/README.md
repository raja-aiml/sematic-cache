# Semantic Cache Deployment

This directory contains the complete deployment solution for the Semantic Cache application using Kubernetes with [k3d](https://k3d.io/) for local development and production-ready manifests.

## Directory Structure

```
deploy/
├── cluster.sh                # Wrapper script for cluster management  
├── dev.sh                   # Wrapper script for development workflow
├── README.md                # This file
├── build/                   # Build artifacts and Dockerfile
│   └── Dockerfile           # Application container build definition
├── config/                  # Kubernetes configuration files
│   ├── app/
│   │   ├── kustomization.yaml    # App kustomization
│   │   └── sematic-cache.yaml    # App deployment, service, ingress
│   └── infra/
│       ├── kustomization.yaml    # Infrastructure kustomization
│       ├── postgres.yaml         # PostgreSQL with pgvector
│       ├── redis.yaml            # Redis cache
│       └── ingress-nginx/
│           ├── kustomization.yaml
│           ├── nginx-web.yaml
│           └── test-ingress-nginx.yaml
├── scripts/                 # Development and utility scripts
│   ├── cluster.sh          # Cluster infrastructure management
│   ├── dev.sh              # Application development workflow
│   └── debug.sh            # Debugging utilities
└── web/                     # Static web content
    └── index.html          # Kubernetes deployment web interface
```

## Prerequisites

- Docker
- [k3d](https://k3d.io/) (tested with v5+)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [HTTPie](https://httpie.io/) (for API testing)

## Organized Structure

The deployment follows enterprise-standard organization:

- **`build/`**: Container build artifacts and Dockerfile
- **`config/`**: Kubernetes manifests organized by component
  - `app/`: Application deployments, services, and ingress
  - `infra/`: Infrastructure components (database, cache, ingress controller)
- **`scripts/`**: Development and utility scripts
- **`web/`**: Static web content served via ingress

This structure provides:
- Clear separation between infrastructure and application concerns
- Scalable organization for complex deployments
- Easy maintenance and configuration management
- Single deployment target eliminates complexity
- Production-ready with development-friendly tooling

## Project Structure

```
k8s/
├── cluster.sh           # Manages k3d cluster lifecycle (create, delete, logs)
├── dev.sh               # Builds and deploys Docker image using k3d import
├── infra/               # Infrastructure components
│   ├── postgres.yaml    # PostgreSQL with pgvector extension
│   ├── redis.yaml       # Redis cache
│   ├── ingress-nginx/   # Ingress controller
│   │   └── kustomization.yaml
│   └── kustomization.yaml
├── app/                 # Application layer
│   ├── sematic-cache.yaml  # Sematic Cache API deployment
│   └── kustomization.yaml
└── README.md
```

## Usage

### 1. Make scripts executable
```bash
chmod +x cluster.sh dev.sh
```

### 2. Create k3d cluster and infrastructure
```bash
deploy/k8s/cluster.sh up
```

This will:
- Create a k3d cluster named `sematic-cache`
- Deploy PostgreSQL with pgvector extension
- Deploy Redis cache
- Deploy Ingress NGINX controller
- Wait for all deployments to be ready

### 3. Build and deploy application
```bash
# Build Docker image and import into k3d cluster
deploy/k8s/dev.sh build

# Deploy the application
deploy/k8s/dev.sh  deploy
```

### 4. Check deployment status
```bash
# View cluster and infrastructure status
deploy/k8s/cluster.sh ps

# View application status
deploy/k8s/cluster.sh test

# Run infrastructure health checks
deploy/k8s/cluster.sh test
```

### 5. View logs
```bash
# View logs for all infrastructure pods
deploy/k8s/cluster.sh logs

# View logs for a specific pod with follow
deploy/k8s/cluster.sh logs <pod-name> --follow
```

### 6. Cleanup
```bash
# Remove the entire k3d cluster
deploy/k8s/cluster.sh down
```

## Accessing the Application

After deployment, the Sematic Cache API is accessible via:

- **LoadBalancer**: `http://localhost:8080`
- **Ingress**: `http://sematic.127.0.0.1.nip.io:8080`
## Components

### Infrastructure (`infra/` namespace)
- **PostgreSQL**: Database with pgvector extension for vector operations
- **Redis**: In-memory cache for fast data retrieval
- **Ingress NGINX**: HTTP/HTTPS ingress controller for routing

### Application (`app/` namespace)
- **Sematic Cache API**: Main application service with REST endpoints

## Architecture Benefits

- **Simple**: No registry complexity - uses k3d's native image import
- **Fast**: Direct image import vs registry push/pull cycle
- **Local**: Everything runs locally with k3d
- **Reproducible**: Consistent deployment using Kustomize

## Development Workflow

```bash
# Start fresh
deploy/k8s/cluster.sh  up

# Make code changes, then rebuild and redeploy
deploy/k8s/dev.sh build
deploy/k8s/dev.sh deploy

# Check status
deploy/k8s/dev.sh test

# View logs if needed
deploy/k8s/cluster.sh  logs

# Clean up when done
deploy/k8s/cluster.sh  down
```

## Configuration

### Database Connection
The application connects to PostgreSQL using:
```
postgres://cache:cache@postgres.infra:5432/cache?sslmode=disable
```

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string
- `OPENAI_API_KEY`: Set to `dummy-key` for local development

## Troubleshooting

### Check pod status
```bash
kubectl get pods -A
```

### View specific deployment logs
```bash
kubectl logs -n app deployment/sematic-cache
kubectl logs -n infra deployment/postgres
kubectl logs -n infra deployment/redis
```

### Restart a deployment
```bash
kubectl rollout restart deployment/sematic-cache -n app
```

### Access services directly
```bash
# Port forward to access services directly
kubectl port-forward -n app svc/sematic-cache 8080:8080
kubectl port-forward -n infra svc/postgres 5432:5432
kubectl port-forward -n infra svc/redis 6379:6379
```

## End-to-End Testing with HTTPie

### Infrastructure Testing

```bash
# Test infrastructure components
deploy/k8s/cluster.sh test

# Test individual services
kubectl get pods -n infra
kubectl get svc -n infra

# Test database connectivity
kubectl exec -n infra deployment/postgres -- psql -U cache -d cache -c "SELECT version();"
kubectl exec -n infra deployment/postgres -- psql -U cache -d cache -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Test Redis connectivity  
kubectl exec -n infra deployment/redis -- redis-cli ping
```

### Application API Testing

```bash
# Health checks
http GET http://localhost:8080/semantic-cache/health
http GET http://localhost:8080/semantic-cache/metrics

# Basic cache operations
http POST http://localhost:8080/semantic-cache/set \
  prompt="What is Kubernetes?" \
  answer="Kubernetes is a container orchestration platform" \
  modelName="gpt-3.5-turbo"

http POST http://localhost:8080/semantic-cache/get \
  prompt="What is Kubernetes?"

# Test similarity search (requires real OpenAI API key)
http POST http://localhost:8080/semantic-cache/get \
  prompt="Tell me about K8s"

# Advanced queries
http POST http://localhost:8080/semantic-cache/topk \
  embedding:='[0.1, 0.2, 0.3]' \
  k:=5

http POST http://localhost:8080/semantic-cache/query \
  embedding:='[0.1, 0.2, 0.3]'
```

### Web Interface Testing

```bash
# Test web interface
http GET http://localhost:8080/web/
curl -s http://localhost:8080/web/ | grep "Kubernetes Deployment"

# Test proxy routing
http GET http://localhost:8080/proxy-health  # Should not exist in k8s (Docker only)
```

### Performance Testing

```bash
# Metrics over time
for i in {1..10}; do
  http POST http://localhost:8080/semantic-cache/set \
    prompt="Test prompt $i" \
    answer="Test answer $i" \
    modelName="test"
  sleep 1
done

# Check final metrics
http GET http://localhost:8080/semantic-cache/metrics

# Load testing with multiple requests
seq 1 50 | xargs -P 5 -I {} \
  http POST http://localhost:8080/semantic-cache/get prompt="What is Kubernetes?"
```

Notes

- PostgreSQL automatically installs the `vector` extension on startup
- Images are imported directly into k3d using `k3d image import`
- No external registry or complex networking required
- Ingress NGINX handles HTTP routing with LoadBalancer integration
- All data is ephemeral - destroyed when cluster is deleted
- HTTPie provides clean JSON output for API testing and debugging
- Single deployment target reduces maintenance overhead and complexity
- Enterprise-ready with production capabilities built-in

## Production Deployment

For production environments:

1. **Replace k3d with production Kubernetes cluster**
2. **Update ingress configuration** for your domain
3. **Configure persistent storage** for PostgreSQL data
4. **Set up proper secrets management** (e.g., sealed-secrets, external-secrets)
5. **Configure resource limits and requests**
6. **Set up monitoring and alerting**

The manifests are designed to be production-ready with minimal modifications.

---

