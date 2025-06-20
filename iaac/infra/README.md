# IaaC - Infrastructure as Code Tool

A generic, production-ready infrastructure-as-code tool for deploying applications to local Kubernetes environments.

## Quick Start

```bash
# Build the tool
task build

# Deploy everything
./bin/iaac workflow full
```

## Overview

This directory contains a generic Go-based CLI tool that manages the complete lifecycle of application infrastructure:

- **Local Kubernetes**: k3d cluster creation and management
- **Infrastructure Components**: PostgreSQL, Redis, and NGINX Ingress
- **Application Deployment**: Automated build, deploy, and configuration
- **Testing**: Built-in endpoint testing and validation
- **Fully Configurable**: Adaptable to any application via environment variables

## Prerequisites

- Go 1.23.8+
- Docker
- k3d v5.x
- kubectl

## Configuration

The tool is fully configurable via environment variables:

```bash
export IAAC_APP_NAME=myapp
export IAAC_CLUSTER_NAME=myapp-dev
export IAAC_IMAGE_NAME=myapp:latest
export IAAC_BLUEPRINT_PATH=deploy/k8s
```

See [Environment Variables Guide](docs/guide/ENV_VARS.md) for complete configuration options.

## Documentation

Comprehensive documentation is available in the `docs/guide` directory:

- [**README.md**](docs/guide/README.md) - Complete command reference and usage guide
- [**DEVELOPMENT_GUIDE.md**](docs/guide/DEVELOPMENT_GUIDE.md) - Development workflow and testing
- [**PRODUCTION_GUIDE.md**](docs/guide/PRODUCTION_GUIDE.md) - Production deployment best practices
- [**ENV_VARS.md**](docs/guide/ENV_VARS.md) - Environment variables configuration

## Key Features

- **SDK Integration**: Uses Docker SDK and Kubernetes client-go for reliable operations
- **Production Workflows**: Automated end-to-end deployment with health checks
- **Composite Backend**: Test three-tier cache architecture (Memory + Redis + PostgreSQL)
- **Debug Tools**: Built-in troubleshooting and secret management
- **Type Safety**: Full Go type safety with comprehensive error handling

## Commands at a Glance

```bash
# Cluster management
iaac cluster up/down/ps/test

# Development workflow
iaac dev build/deploy/test/logs

# Production workflow
iaac workflow full/reset

# Debugging
iaac debug analyze full
iaac debug secrets create

# Composite testing
iaac composite-test
```

## Project Structure

```
iaac/infra/
├── cmd/              # Command implementations
├── pkg/              # Core packages (k3d, kubernetes, docker, etc.)
├── config/           # Configuration management
├── docs/guide/       # Documentation
├── bin/              # Compiled binaries (git-ignored)
└── Taskfile.yaml     # Build automation
```

## Adapting to Your Project

1. Set environment variables for your application:
   ```bash
   export IAAC_APP_NAME=yourapp
   export IAAC_IMAGE_NAME=yourapp:latest
   export IAAC_BLUEPRINT_PATH=path/to/k8s/manifests
   ```

2. Create your Kubernetes manifests following the blueprint structure:
   ```
   your-blueprint-path/
   ├── infra/
   │   ├── base/
   │   └── overlays/
   └── app/
       ├── base/
       └── overlays/
   ```

3. Run the deployment:
   ```bash
   iaac workflow full
   ```

## License

This is an open-source infrastructure tool. See LICENSE for details.