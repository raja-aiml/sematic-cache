# Infrastructure as Code (iaac)

A comprehensive Infrastructure as Code solution for deploying and managing applications on Kubernetes using k3d.

## Directory Structure

```
iaac/
├── README.md          # This file
├── config/            # Centralized configuration directory
│   ├── blueprint.env  # Environment configuration
│   ├── agent.yaml     # NLP agent configuration
│   └── deploy-config.yaml  # Deployment settings
├── blueprint/         # Kubernetes manifests and configurations
│   ├── app/          # Application manifests
│   ├── services/     # Service definitions (PostgreSQL, Redis)
│   └── infra/        # Infrastructure components
└── infra/            # Go-based CLI tool
    ├── cmd/          # CLI commands
    ├── pkg/          # Core packages
    └── main.go       # Entry point
```

## Quick Start

### Prerequisites

- Docker installed and running
- Go 1.23+ (for building from source)
- kubectl (optional, for direct cluster interaction)

### Installation

1. **Build the CLI tool**:
   ```bash
   cd iaac/infra
   go build -o iaac .
   ```

2. **Set up configuration**:
   ```bash
   cd ../..
   cp iaac/config/blueprint.env.example iaac/config/blueprint.env
   # Edit blueprint.env with your settings
   ```

3. **Create cluster and deploy**:
   ```bash
   # Using default config directory
   ./iaac/infra/iaac cluster up
   
   # Or specify custom config
   ./iaac/infra/iaac --config-dir /path/to/config cluster up
   ```

## Configuration Management

### Config Directory

All configuration files are centralized in the `config/` directory:

- **blueprint.env**: Environment variables for cluster and services
- **agent.yaml**: Configuration for the NLP agent
- **deploy-config.yaml**: Kubernetes deployment parameters

### Using Configuration

1. **Default behavior**: The tool looks for config in these locations:
   - `./config` (current directory)
   - `../config` (parent directory)
   - `../../iaac/config` (when running from subdirectories)

2. **Custom config directory**:
   ```bash
   iaac --config-dir /custom/path cluster up
   ```

3. **Specific environment file**:
   ```bash
   iaac --env-file /custom/blueprint.env cluster up
   ```

## Core Features

### Cluster Management

```bash
# Create cluster with minimal blueprint
iaac cluster up --scenario minimal

# Create with full stack
iaac cluster up --scenario full-stack

# Destroy cluster
iaac cluster down

# Check status
iaac cluster ps
```

### Application Development

```bash
# Build and import Docker image
iaac dev build

# Deploy application
iaac dev deploy

# View logs
iaac dev logs

# Run tests
iaac dev test
```

### Natural Language Interface

```bash
# Single command
iaac agent "create a new cluster"

# Interactive mode
iaac agent --interactive

# With custom config
iaac agent --config config/agent.yaml "deploy my app"
```

## Blueprint Scenarios

The tool supports multiple pre-configured scenarios:

1. **minimal**: Basic PostgreSQL and Redis setup
2. **development**: Full dev stack with debugging tools
3. **service-mesh**: Istio with observability
4. **monitoring-only**: Just the observability stack
5. **full-stack**: Complete production-like environment

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

## Advanced Usage

### Custom Manifests

Deploy custom Kubernetes manifests:

```bash
iaac manifest generate --scenario custom --path /path/to/manifests
iaac manifest render --path /path/to/kustomization
```

### Workflow Automation

Run complete workflows:

```bash
# Full deployment workflow
iaac workflow full

# Just infrastructure setup
iaac workflow setup

# Application deployment only
iaac workflow deploy
```

### Validation and Testing

```bash
# Validate blueprint structure
iaac validate blueprint

# Validate deployed resources
iaac validate deployment

# Run integration tests
iaac test --scenario minimal
```

## Troubleshooting

### Common Issues

1. **Config not found**:
   ```bash
   # Explicitly specify config directory
   iaac --config-dir $(pwd)/iaac/config cluster up
   ```

2. **Docker permissions**:
   ```bash
   # Ensure Docker daemon is accessible
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

3. **Port conflicts**:
   ```bash
   # Check for existing k3d clusters
   k3d cluster list
   k3d cluster delete <name>
   ```

### Debug Mode

Enable detailed logging:

```bash
# Set log level
export LOG_LEVEL=debug

# Or use debug command
iaac debug analyze full
```

## Best Practices

1. **Configuration Management**:
   - Keep sensitive data in `.env` files (not committed)
   - Use `.env.example` as templates
   - Store environment-specific configs separately

2. **Version Control**:
   - Commit blueprint manifests
   - Commit example configs only
   - Use `.gitignore` for actual config files

3. **Development Workflow**:
   - Use `minimal` scenario for quick testing
   - Use `development` for feature development
   - Use `full-stack` for production simulation

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](../LICENSE) for details.