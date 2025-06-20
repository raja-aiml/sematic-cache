# Monitoring-Only Scenario

Pure observability stack without application infrastructure.

## What's Included

- **Metrics**: Prometheus + Alertmanager
- **Visualization**: Grafana with pre-configured dashboards
- **Logging**: Loki + Fluent Bit + Promtail
- **Tracing**: OpenTelemetry Collector + Tempo + Jaeger UI

## Use Cases

This scenario is perfect for:
- Adding observability to existing clusters
- Monitoring external services
- Development of monitoring solutions
- Learning observability tools

## Quick Start

```bash
# Deploy monitoring stack
kubectl apply -k scenarios/monitoring-only

# Port-forward services
kubectl port-forward -n monitoring svc/grafana 3000:3000 &
kubectl port-forward -n monitoring svc/prometheus 9090:9090 &
kubectl port-forward -n monitoring svc/alertmanager 9093:9093 &
kubectl port-forward -n logging svc/loki 3100:3100 &
kubectl port-forward -n tracing svc/otel-visualizer 16686:16686 &

# Access UIs
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# Alertmanager: http://localhost:9093
# Jaeger: http://localhost:16686
```

## Configuration

### Adding External Targets

Add external services to monitor in Prometheus:

```yaml
- job_name: 'external-service'
  static_configs:
  - targets: ['external.example.com:9090']
    labels:
      environment: 'production'
      service: 'external-api'
```

### Custom Dashboards

Import dashboards from grafana.com or create custom ones:

1. Access Grafana UI
2. Navigate to Dashboards > Import
3. Enter dashboard ID or upload JSON

### Log Sources

Configure additional log sources in Fluent Bit:

```yaml
[INPUT]
    Name tail
    Path /var/log/custom/*.log
    Tag custom.*
```

## Storage Considerations

Default storage allocations:
- Prometheus: 20GB (15 days retention)
- Loki: 20GB (7 days retention)
- Grafana: 5GB
- Tempo: 10GB (24 hours retention)

Adjust these in `values.yaml` based on your needs.