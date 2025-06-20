# Service Mesh Scenario

Istio service mesh deployment with enhanced observability and security.

## What's Included

- Istio control plane with demo profile
- Automatic sidecar injection
- mTLS enabled by default
- Enhanced telemetry collection
- Service mesh specific dashboards
- Traffic management policies

## Quick Start

```bash
# Deploy the service mesh scenario
kubectl apply -k scenarios/service-mesh

# Install Istio (if not already installed)
istioctl install --set profile=demo -y

# Enable sidecar injection for namespaces
kubectl label namespace infra istio-injection=enabled
kubectl label namespace app istio-injection=enabled

# Access Istio dashboards
istioctl dashboard kiali
istioctl dashboard grafana
istioctl dashboard jaeger
```

## Features

### Traffic Management
- Intelligent routing
- Load balancing
- Circuit breaking
- Retry policies
- Timeouts

### Security
- Automatic mTLS
- Authorization policies
- JWT authentication
- Rate limiting

### Observability
- Distributed tracing
- Service metrics
- Service graph visualization
- Performance monitoring

## Configuration Examples

### Virtual Service
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: postgres-vs
spec:
  hosts:
  - postgres.infra.svc.cluster.local
  http:
  - timeout: 30s
    retries:
      attempts: 3
      perTryTimeout: 10s
```

### Destination Rule
```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: postgres-dr
spec:
  host: postgres.infra.svc.cluster.local
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
```