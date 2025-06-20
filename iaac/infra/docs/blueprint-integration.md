# Blueprint Integration Guide

This document describes how `iaac/infra` has been refactored to utilize the blueprint components from `iaac/blueprint`.

## Overview

The `iaac/infra` tool now leverages the comprehensive K3D Blueprint system for deploying infrastructure components. This integration provides:

- **Scenario-based deployments**: Choose from pre-configured scenarios (minimal, development, service-mesh, monitoring-only, full-stack)
- **Modular architecture**: Components are organized as reusable modules
- **Environment overlays**: Support for different environments (local, dev)
- **Comprehensive observability**: Built-in monitoring, logging, and tracing
- **Service mesh integration**: Optional Istio deployment with security policies

## Changes Made

### 1. Updated Constants (`pkg/constants/constants.go`)

Added blueprint-specific constants:
- Blueprint paths (base, infra, scenarios, modules)
- Scenario names (minimal, development, service-mesh, monitoring-only, full-stack)
- Additional namespaces (monitoring, istio-system, logging, tracing)
- Helper functions for path construction

### 2. Enhanced Cluster Command (`cmd/cluster.go`)

The cluster command now supports:
- `--scenario` flag: Select blueprint scenario to deploy
- `--overlay` flag: Choose environment overlay (local, dev)
- `--kustomize-path` flag: Still supported for custom configurations

New functionality:
- Automatic blueprint path detection
- Scenario-based component waiting
- Enhanced status display for all namespaces
- Blueprint validation test integration

### 3. Updated Workflow Command (`cmd/workflow.go`)

The workflow orchestrator now:
- Defaults to "development" scenario for complete workflows
- Supports scenario selection via `--scenario` flag
- Adjusts completion messages based on deployed scenario

## Usage Examples

### Deploy Minimal Infrastructure
```bash
# Deploy just PostgreSQL and Redis
bin/iaac cluster up --scenario minimal

# Deploy with dev overlay (more resources)
bin/iaac cluster up --scenario minimal --overlay dev
```

### Deploy Development Environment
```bash
# Full development stack with observability
bin/iaac cluster up --scenario development

# Run workflow with development scenario
bin/iaac workflow full --scenario development
```

### Deploy Service Mesh
```bash
# Istio with mTLS and traffic management
bin/iaac cluster up --scenario service-mesh
```

### Deploy Monitoring Stack Only
```bash
# Just the observability components
bin/iaac cluster up --scenario monitoring-only
```

### Deploy Full Production-like Stack
```bash
# Everything: infra, service mesh, observability
bin/iaac cluster up --scenario full-stack
```

### Custom Kustomize Path (Legacy Support)
```bash
# Still supports direct kustomize paths
bin/iaac cluster up --kustomize-path /path/to/kustomization
```

## Scenario Details

### Minimal (`minimal`)
- PostgreSQL with pgvector
- Redis
- Basic networking

### Development (`development`)
- Everything from minimal
- Prometheus & Grafana
- Loki for logging
- Debug tools
- Resource limits optimized for development

### Service Mesh (`service-mesh`)
- Istio control plane
- Ingress/Egress gateways
- mTLS enforcement
- Traffic policies
- Observability integration

### Monitoring Only (`monitoring-only`)
- Prometheus with alerting
- Grafana with dashboards
- Loki & Fluent Bit
- OpenTelemetry & Tempo
- No application infrastructure

### Full Stack (`full-stack`)
- All components from all scenarios
- Production-like configuration
- Complete observability
- Service mesh enabled
- Security policies enforced

## Testing

The cluster command now integrates with the blueprint validation kit:

```bash
# Run tests for deployed scenario
bin/iaac cluster test

# Tests include:
# - Connectivity tests
# - Blueprint validation suite
# - Scenario-specific tests
```

## Migration from Old Structure

If you have existing deployments:

1. The old kustomize structure in `iaac/infra/kustomize/` is still supported
2. Use `--kustomize-path` to reference old configurations
3. Gradually migrate to blueprint scenarios for better maintainability

## Benefits of Blueprint Integration

1. **Consistency**: All components follow the same patterns
2. **Modularity**: Easy to add/remove components
3. **Testability**: Comprehensive validation suite included
4. **Documentation**: Each component is well-documented
5. **Best Practices**: Security, observability, and operational excellence built-in
6. **Flexibility**: Mix and match components via scenarios

## Next Steps

1. Test each scenario in your environment
2. Customize overlays for your specific needs
3. Extend scenarios with your own components
4. Contribute improvements back to the blueprint

## Troubleshooting

### Scenario Not Found
Ensure the `iaac/blueprint/scenarios/<scenario-name>` directory exists.

### Component Not Ready
Check logs with:
```bash
bin/iaac cluster logs -n <namespace>
```

### Custom Configuration Needed
Create a new scenario or use `--kustomize-path` for one-off deployments.