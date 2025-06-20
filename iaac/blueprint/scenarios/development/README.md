# Development Scenario

Full development environment with all observability features enabled.

## What's Included

- PostgreSQL with pgvector extension
- Redis with persistence
- Full observability stack:
  - Prometheus + Alertmanager
  - Grafana with pre-configured dashboards
  - Loki + Fluent Bit for logging
  - OpenTelemetry + Tempo for tracing
- Development tools and debug utilities
- Hot-reload enabled
- Debug endpoints exposed

## Quick Start

```bash
# Deploy the development scenario
kubectl apply -k scenarios/development

# Port-forward services for local access
kubectl port-forward -n monitoring svc/grafana 3000:3000 &
kubectl port-forward -n monitoring svc/prometheus 9090:9090 &
kubectl port-forward -n tracing svc/otel-visualizer 16686:16686 &

# Access services
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# Jaeger: http://localhost:16686
```

## Configuration

The development scenario includes:

1. **Enhanced Resource Limits**: More generous CPU/memory allocations
2. **Debug Tools**: Additional pods for troubleshooting
3. **Profiling Enabled**: Application profiling endpoints
4. **Verbose Logging**: Debug-level logging across all components
5. **Trace Sampling**: 100% trace sampling for detailed debugging

## Customization

To customize this scenario, modify:
- `values.yaml`: Override default values
- `dev-tools.yaml`: Add or remove debug tools
- `kustomization.yaml`: Include/exclude modules