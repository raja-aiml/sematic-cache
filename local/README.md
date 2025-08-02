# Semantic Cache Observability Stack

This directory contains the complete observability stack for the Semantic Cache application.

## Architecture

```
Application → OTel Collector → Jaeger (Traces)
                            → Prometheus (Metrics)
                            
Caddy (Port 80) → Unified Web Interface → All Services
```

## Components

- **PostgreSQL with pgvector**: Vector database for semantic similarity search
- **Redis**: L2 cache tier for fast lookups
- **OpenTelemetry Collector**: Central telemetry hub that receives, processes, and exports traces and metrics
- **Jaeger**: Distributed tracing backend for visualizing request flows
- **Prometheus**: Time-series database for metrics collection and alerting
- **Caddy**: Web server providing unified access to all services via port 80

## Quick Start

1. Start the stack:
```bash
docker compose up -d
```

2. Set your OpenAI API key and run the application:
```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run from project root
go run cmd/server.go -config local/config.yaml

# Or if you built the binary
./bin/semantic-cache -config local/config.yaml
```

3. Access everything through Caddy (recommended):
- **Dashboard**: http://localhost
- **Jaeger UI**: http://localhost/jaeger
- **Prometheus**: http://localhost/prometheus
- **OTel Health**: http://localhost/otel/health
- **OTel ZPages**: http://localhost/otel/zpages
- **Service Status**: http://localhost/api/status

Or access services directly:
- Jaeger UI: http://localhost:16686
- Prometheus UI: http://localhost:9090
- OTel Collector Health: http://localhost:13133/health
- OTel Collector ZPages: http://localhost:55679

4. Configure your application to send telemetry to the OTel Collector:
```yaml
observability:
  otel:
    endpoint: localhost:4317  # OTel Collector gRPC endpoint
```

## Ports

| Service | Port | Description |
|---------|------|-------------|
| **Caddy** | 80 | Unified web interface (dashboard + reverse proxy) |
| **Caddy** | 443 | HTTPS (auto-cert if domain configured) |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache |
| OTel Collector | 4317 | OTLP gRPC receiver |
| OTel Collector | 4318 | OTLP HTTP receiver |
| OTel Collector | 8888 | Collector metrics |
| OTel Collector | 8889 | Prometheus exporter |
| OTel Collector | 13133 | Health check |
| OTel Collector | 55679 | ZPages (debug) |
| Jaeger | 16686 | Web UI |
| Jaeger | 14250 | gRPC model.proto |
| Prometheus | 9090 | Web UI |

## Configuration Files

- `docker-compose.yml`: Docker Compose configuration
- `otel-collector-config.yaml`: OpenTelemetry Collector configuration
- `prometheus.yaml`: Prometheus scrape configuration
- `init-pgvector.sql`: PostgreSQL initialization script
- `config.yaml`: Application configuration (ready to use)

## Monitoring

### Traces
View distributed traces in Jaeger UI at http://localhost:16686

### Metrics
Query metrics in Prometheus at http://localhost:9090

Example queries:
- `rate(cache_hits_total[5m])` - Cache hit rate
- `histogram_quantile(0.95, rate(query_duration_bucket[5m]))` - 95th percentile query latency

## Troubleshooting

Check service health:
```bash
docker compose ps
docker compose logs otel-collector
docker compose logs jaeger
```

Verify OTel Collector is receiving data:
```bash
curl http://localhost:13133/health
```

## Cleanup

Stop and remove all containers:
```bash
docker compose down -v
```