# Semantic Cache Infrastructure

Production-ready infrastructure-as-code for deploying the Semantic Cache system to local Kubernetes environments.

## Quick Start

```bash
# Build the tool
task build

# Deploy everything
./bin/iaac workflow full
```

## Overview

This directory contains a Go-based CLI tool that manages the complete lifecycle of the Semantic Cache infrastructure:

- **Local Kubernetes**: k3d cluster creation and management
- **Infrastructure Components**: PostgreSQL with pgvector, Redis, and NGINX Ingress
- **Application Deployment**: Automated build, deploy, and configuration
- **Testing**: Built-in endpoint testing and composite backend validation

## Prerequisites

- Go 1.23.8+
- Docker
- k3d v5.x
- kubectl

## Documentation

Comprehensive documentation is available in the `docs/guide` directory:

- [**README.md**](docs/guide/README.md) - Complete command reference and usage guide
- [**DEVELOPMENT_GUIDE.md**](docs/guide/DEVELOPMENT_GUIDE.md) - Development workflow and testing
- [**PRODUCTION_GUIDE.md**](docs/guide/PRODUCTION_GUIDE.md) - Production deployment best practices

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

## License

Part of the Semantic Cache project. See the main repository for license information.