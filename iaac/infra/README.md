# Infrastructure as Code CLI Tool

A comprehensive CLI tool for managing application deployments in local Kubernetes environments with natural language support.

## Features

- **K3D Cluster Management**: Create, manage, and destroy local Kubernetes clusters
- **Application Deployment**: Deploy and manage applications with Docker and Kubernetes
- **Natural Language Interface**: Use plain English to execute infrastructure commands
- **Infrastructure Validation**: Validate configurations and deployments
- **Workflow Automation**: Automate complex deployment workflows

## Installation

### From Source

```bash
go build -o iaac .
```

### Using Go Install

```bash
go install github.com/raja-aiml/sematic-cache/deploy/local@latest
```

## Configuration

The tool uses a centralized configuration directory. By default, it looks for configuration in:
- `./config`
- `../config`
- `../../iaac/config`

### Global Flags

```bash
--config-dir string   Config directory path (default: ./config or ../config)
--env-file string     Path to environment file (default: <config-dir>/blueprint.env)
```

### Examples

```bash
# Use default config location
iaac cluster up

# Use custom config directory
iaac --config-dir /path/to/config cluster up

# Use specific environment file
iaac --env-file /path/to/custom.env cluster up

# Check current configuration
iaac config show
```

## Commands

### Cluster Management

Manage k3d clusters with blueprint scenarios.

```bash
# Create cluster with minimal scenario
iaac cluster up --scenario minimal

# Create cluster with full stack
iaac cluster up --scenario full-stack

# Destroy cluster
iaac cluster down

# Check cluster status
iaac cluster ps

# View cluster logs
iaac cluster logs

# Run cluster tests
iaac cluster test
```

#### Available Scenarios

- `minimal`: Basic PostgreSQL and Redis
- `development`: Full development stack with debug tools
- `service-mesh`: Istio service mesh with observability
- `monitoring-only`: Just the observability stack
- `full-stack`: Complete production-like environment

### Development Commands

Build and deploy applications.

```bash
# Build Docker image and import to k3d
iaac dev build

# Deploy application with secrets
iaac dev deploy

# View application logs
iaac dev logs --follow

# Check deployment status
iaac dev status

# Run application tests
iaac dev test

# Remove deployment
iaac dev remove
```

### Natural Language Agent

Use natural language to execute commands.

```bash
# Single query
iaac agent "create a new cluster with 3 nodes"

# Interactive mode
iaac agent --interactive

# With custom configuration
iaac agent --config config/agent.yaml "deploy my application"
```

#### Agent Configuration

Create `config/agent.yaml`:

```yaml
openai_key: ${OPENAI_API_KEY}
openai_model: gpt-4-turbo-preview
openai_max_tokens: 1000

enable_dangerous_commands: false
require_confirmation: true

command_whitelist:
  - cluster
  - dev
  - workflow

command_blacklist:
  - "cluster delete"
  - "dev remove"
```

### Workflow Commands

Automate complex deployment workflows.

```bash
# Full deployment workflow
iaac workflow full

# Setup infrastructure only
iaac workflow setup

# Build application only
iaac workflow build

# Deploy application only
iaac workflow deploy

# Run tests
iaac workflow test

# Show workflow status
iaac workflow status

# Clean up everything
iaac workflow reset
```

### Manifest Management

Generate and manage Kubernetes manifests.

```bash
# Generate manifests for a scenario
iaac manifest generate --scenario minimal

# Generate with custom overlay
iaac manifest generate --overlay dev

# Render manifests with environment substitution
iaac manifest render --path ./manifests

# Show manifest differences
iaac manifest diff --scenario minimal
```

### Validation Commands

Validate configurations and deployments.

```bash
# Validate blueprint structure
iaac validate blueprint --path ./blueprint

# Validate deployed resources
iaac validate deployment --namespace app

# Validate manifest files
iaac validate manifests --path ./manifests
```

### Test Commands

Run validation tests.

```bash
# Run tests for current scenario
iaac test --scenario minimal

# Run tests in parallel
iaac test --parallel

# Generate test report
iaac test --report json --output report.json

# Run with custom timeout
iaac test --timeout 600
```

### Configuration Commands

Manage configuration.

```bash
# Show current configuration
iaac config show

# Get configuration directory path
iaac config path
```

### Debug Commands

Comprehensive debugging tools.

```bash
# Full diagnostic analysis
iaac debug analyze full

# Quick analysis
iaac debug analyze quick

# Test API endpoints
iaac debug test quick

# Manage secrets
iaac debug secrets create
iaac debug secrets update
iaac debug secrets view
```

## Environment Variables

Key environment variables (set in `config/blueprint.env`):

```bash
# Cluster Configuration
K3D_CLUSTER_NAME=semantic-cache
K3D_K3S_VERSION=v1.31.5-k3s1
K3D_NODE_COUNT=3

# Application Settings
APP_VERSION=latest
APP_PORT=8080

# Database Configuration
DB_NAME=cache_db
DB_PASSWORD=your_password

# Redis Configuration
REDIS_PASSWORD=your_redis_password
```

## Documentation Generation

Generate command documentation for the NLP agent:

```bash
# Generate JSON documentation
iaac docs --output commands.json

# Generate Markdown documentation
iaac docs --format markdown --output commands.md
```

## Examples

### Quick Start

```bash
# Set up configuration
cp config/blueprint.env.example config/blueprint.env
# Edit blueprint.env with your settings

# Create cluster and deploy
iaac workflow full

# Or step by step
iaac cluster up
iaac dev build
iaac dev deploy
iaac dev test
```

### Using Natural Language

```bash
# Start interactive agent
iaac agent -i

# Example queries:
> create a new cluster
> deploy my application
> show all running pods
> run tests
```

### Custom Scenarios

```bash
# Deploy with custom manifests
iaac manifest generate --path ./my-manifests
iaac cluster up --kustomize-path ./my-manifests
```

## Troubleshooting

### Configuration Issues

```bash
# Check configuration paths
iaac config show

# Use explicit paths
iaac --config-dir $(pwd)/config cluster up
```

### Docker Issues

```bash
# Ensure Docker is running
docker ps

# Check Docker permissions
docker run hello-world
```

### Cluster Issues

```bash
# List existing clusters
k3d cluster list

# Delete conflicting cluster
k3d cluster delete semantic-cache
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=debug
iaac cluster up

# Or use debug commands
iaac debug analyze full
```

## Project Structure

```
iaac/infra/
├── cmd/              # CLI commands
├── pkg/              # Core packages
│   ├── agent/       # NLP agent
│   ├── blueprint/   # Blueprint management
│   ├── config/      # Configuration loader
│   ├── docker/      # Docker operations
│   ├── k3d/         # K3D cluster management
│   ├── kubernetes/  # Kubernetes client
│   ├── llm/         # LLM integration
│   ├── secrets/     # Secret management
│   └── utils/       # Utilities
├── config/          # Default configuration
├── docs/            # Documentation
└── main.go          # Entry point
```

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

## License

This project is licensed under the MIT License.