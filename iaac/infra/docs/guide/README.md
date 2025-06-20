# Semantic Cache Infrastructure - Go Implementation

A production-ready Go-based CLI tool for managing semantic cache deployments in local Kubernetes environments using k3d. This tool provides a complete infrastructure-as-code solution with Docker SDK integration, Kubernetes client-go, and comprehensive testing capabilities.

## Overview

This tool replaces traditional shell scripts with a robust Go implementation that provides:
- Type-safe configuration management
- Comprehensive error handling
- SDK-based integrations (Docker SDK, Kubernetes client-go)
- Production-ready deployment workflows
- Built-in testing and debugging capabilities

## Features

- **k3d Cluster Management**: Create and manage local Kubernetes clusters
- **Docker SDK Integration**: Build, tag, and manage container images
- **Kubernetes Native**: Direct client-go integration for resource management
- **Three-Tier Cache Testing**: Support for memory + Redis + PostgreSQL backends
- **Production Workflows**: End-to-end deployment automation
- **Comprehensive Debugging**: Built-in troubleshooting and analysis tools

## Prerequisites

- Go 1.23.8 or later
- Docker Desktop or Docker Engine
- k3d v5.x (`brew install k3d` on macOS)
- kubectl (`brew install kubectl` on macOS)
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)

## Installation

```bash
# From the iaac/infra directory
go build -o bin/iaac .

# Or install globally
go install .

# Using Task (recommended)
task build
```

## Quick Start

```bash
# Complete deployment workflow
iaac workflow full

# Step-by-step approach
iaac cluster up        # Create cluster
iaac dev build         # Build image
iaac dev deploy        # Deploy application
iaac dev test          # Test endpoints
```

## Command Reference

### Cluster Management (`cluster`)

Manage k3d clusters with pre-configured infrastructure components.

```bash
# Create cluster with infrastructure
iaac cluster up

# Destroy cluster
iaac cluster down

# Show cluster status
iaac cluster ps

# View pod logs
iaac cluster logs -n app -l app=semantic-cache

# Verify deployment health
iaac cluster test
```

### Development Commands (`dev`)

Build, deploy, and manage the semantic cache application.

```bash
# Build Docker image
iaac dev build

# Deploy with secrets
iaac dev deploy

# Test endpoints
iaac dev test

# View logs
iaac dev logs -f --tail=100

# Check status
iaac dev status

# Remove deployment
iaac dev remove
```

### Workflow Orchestration (`workflow`)

Production-ready deployment workflows with automated testing.

```bash
# Full deployment (setup → build → deploy → test)
iaac workflow full

# Individual phases
iaac workflow setup
iaac workflow build
iaac workflow deploy
iaac workflow test

# Cleanup and reset
iaac workflow cleanup
iaac workflow reset
```

### Composite Backend Testing (`composite-test`)

Test the three-tier cache architecture with cluster services.

```bash
# Run composite backend test
iaac composite-test
```

This command:
- Sets up port forwarding for PostgreSQL and Redis
- Initializes the database with pgvector
- Creates a composite configuration
- Runs the server with all three cache tiers
- Tests the endpoints
- Cleans up automatically

### Debug Commands (`debug`)

Comprehensive debugging and troubleshooting tools.

```bash
# Secret management
iaac debug secrets create
iaac debug secrets view
iaac debug secrets update

# Deployment analysis
iaac debug analyze full
iaac debug analyze quick

# API testing
iaac debug test quick
iaac debug test detailed
```

## Architecture

```
iaac/infra/
├── cmd/                    # Command implementations
│   ├── cluster.go         # Cluster management
│   ├── composite.go       # Composite backend testing
│   ├── debug.go           # Debugging utilities
│   ├── dev.go            # Development workflow
│   └── workflow.go       # Production workflows
├── pkg/                   # Core packages
│   ├── k3d/              # k3d cluster operations
│   ├── kubernetes/       # Kubernetes client operations
│   ├── docker/          # Docker SDK integration
│   ├── secrets/         # Secret management
│   ├── testing/         # HTTP endpoint testing
│   ├── database/        # PostgreSQL utilities
│   ├── constants/       # Configuration constants
│   └── utils/           # Common utilities
├── config/               # Configuration management
├── internal/             # Internal packages
└── docs/guide/          # Documentation

```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI API access
- `DATABASE_URL`: PostgreSQL connection string (auto-configured for cluster)
- `DEBUG`: Enable debug logging
- `SC_DEPLOY_DEBUG`: Enable deployment debug mode

### Configuration File

The tool supports a YAML configuration file (`deploy-config.yaml`) for advanced settings:

```yaml
cluster:
  name: semantic-cache
  api_port: "6550"
  http_port: "8080:80"
  timeout: 5m

build:
  image_name: semantic-cache:local
  dockerfile: Dockerfile
  platform: linux/amd64

deploy:
  namespace: app
  timeout: 5m
  wait_for_ready: true
```

## Key Technologies

- **Go 1.23.8**: Modern Go with enhanced performance
- **Docker SDK**: Native Docker API integration
- **Kubernetes client-go**: Official Kubernetes Go client
- **k3d**: Lightweight Kubernetes for local development
- **Cobra**: CLI framework for command structure
- **OpenTelemetry**: Observability and tracing support

## Documentation

- [Development Guide](DEVELOPMENT_GUIDE.md) - Development workflow, testing, and debugging
- [Production Guide](PRODUCTION_GUIDE.md) - Production deployment and best practices

## Common Workflows

### 1. Fresh Development Setup

```bash
# Start fresh
iaac workflow reset
iaac workflow full
```

### 2. Iterative Development

```bash
# Make code changes, then:
iaac dev build
iaac dev deploy
iaac dev test
iaac dev logs -f
```

### 3. Debugging Issues

```bash
# Quick diagnosis
iaac debug analyze quick

# Full analysis
iaac debug analyze full

# Check secrets
iaac debug secrets view
```

### 4. Testing Composite Backend

```bash
# Test all three cache tiers
iaac composite-test
```

## Troubleshooting

### Common Issues

1. **Cluster creation fails**
   ```bash
   # Check Docker is running
   docker ps
   
   # Check k3d installation
   k3d version
   ```

2. **Build fails**
   ```bash
   # Check Docker daemon
   docker info
   
   # Clean build
   task clean build
   ```

3. **Deployment fails**
   ```bash
   # Check cluster status
   iaac cluster ps
   
   # Analyze deployment
   iaac debug analyze full
   ```

4. **Missing secrets**
   ```bash
   # Create from environment
   iaac debug secrets create
   ```

## Contributing

See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for detailed development instructions.

## License

This project is part of the semantic cache system. See the main project for license details.