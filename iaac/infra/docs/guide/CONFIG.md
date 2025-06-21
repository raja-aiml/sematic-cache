# Configuration Guide

This guide explains how the iaac tool manages configuration files and environment variables.

## Overview

All configuration files are centralized in the `iaac/config` directory for better organization and management.

## Configuration Directory Structure

```
iaac/
└── config/                      # Centralized configuration
    ├── README.md               # Configuration documentation
    ├── blueprint.env           # Main environment configuration
    ├── blueprint.env.example   # Template for blueprint.env
    ├── agent.yaml              # NLP agent configuration
    ├── agent.yaml.example      # Template for agent.yaml
    └── deploy-config.yaml      # Deployment configuration
```

## Using Configuration

### 1. Default Behavior

The tool automatically searches for configuration in these locations (in order):
- `./config`
- `../config`
- `../../iaac/config`

```bash
# From iaac/infra directory
./iaac cluster up
# Automatically uses ../config/blueprint.env
```

### 2. Custom Config Directory

Specify a custom configuration directory:

```bash
iaac --config-dir /custom/path cluster up
```

### 3. Specific Environment File

Use a specific environment file:

```bash
iaac --env-file /custom/blueprint.env cluster up
```

### 4. Check Configuration

View current configuration paths:

```bash
# Show full configuration details
iaac config show

# Get just the config directory path
iaac config path
```

## Environment Files

### blueprint.env

Main configuration file containing:

```bash
# Cluster Configuration
K3D_CLUSTER_NAME=semantic-cache
K3D_K3S_VERSION=v1.31.5-k3s1
K3D_NODE_COUNT=3

# Application Configuration
APP_NAME=semantic-cache
APP_VERSION=latest
APP_PORT=8080
APP_REPLICAS=2

# Database Configuration
DB_NAME=cache_db
DB_USER=postgres
DB_PASSWORD=your_secure_password
DB_PORT=5432

# Redis Configuration
REDIS_PASSWORD=your_redis_password
REDIS_PORT=6379

# Monitoring Configuration
GRAFANA_ADMIN_PASSWORD=admin_password
PROMETHEUS_RETENTION=7d
```

### agent.yaml

Configuration for the NLP agent:

```yaml
# OpenAI Configuration
openai_key: ${OPENAI_API_KEY}
openai_model: gpt-4-turbo-preview
openai_max_tokens: 1000

# Safety Configuration
enable_dangerous_commands: false
require_confirmation: true

# Command Restrictions
command_whitelist:
  - cluster
  - dev
  - workflow

command_blacklist:
  - "cluster delete"
  - "dev remove"

# Execution Settings
command_timeout: 2m
audit_log_path: ./logs/audit.log
```

## Command Examples

### Basic Usage

```bash
# Create cluster using default config
iaac cluster up

# Build application
iaac dev build

# Deploy with custom config
iaac --config-dir ~/myproject/config dev deploy
```

### Environment Overrides

Environment variables override file values:

```bash
# Override cluster name
K3D_CLUSTER_NAME=test-cluster iaac cluster up

# Override multiple values
DB_PASSWORD=newpass REDIS_PASSWORD=redispass iaac dev deploy
```

### Config Directory Resolution

```bash
# Check which config directory will be used
iaac config show

# Output:
# Configuration:
#   Config Directory: /Users/you/project/iaac/config
#   Environment File: /Users/you/project/iaac/config/blueprint.env
#   Config Dir Status: ✓ Exists
#   Env File Status: ✓ Exists
```

## Best Practices

### 1. Security

- Never commit actual `.env` files with secrets
- Use `.env.example` files as templates
- Store production secrets in secure vaults
- Use strong, unique passwords

### 2. Organization

```
config/
├── blueprint.env          # Main config (gitignored)
├── blueprint.env.example  # Template (committed)
├── environments/          # Environment-specific configs
│   ├── local.env
│   ├── dev.env
│   └── prod.env
└── secrets/              # Sensitive configs (gitignored)
    └── api-keys.env
```

### 3. Version Control

Add to `.gitignore`:

```gitignore
# Configuration files with secrets
config/*.env
config/secrets/
!config/*.env.example
```

### 4. Multiple Environments

Use different config directories:

```bash
# Development
iaac --config-dir ./config/dev cluster up

# Staging
iaac --config-dir ./config/staging cluster up

# Production
iaac --config-dir ./config/prod cluster up
```

## Troubleshooting

### Config Not Found

```bash
# Check current working directory
pwd

# Show resolved config path
iaac config show

# Use absolute path
iaac --config-dir $(pwd)/iaac/config cluster up
```

### Environment Variable Issues

```bash
# Check if variable is set
echo $OPENAI_API_KEY

# Export variables from file
export $(cat iaac/config/blueprint.env | xargs)

# Or source the file (if using bash syntax)
source iaac/config/blueprint.env
```

### Permission Issues

```bash
# Check file permissions
ls -la iaac/config/

# Fix permissions if needed
chmod 600 iaac/config/blueprint.env
```

## Migration from Old Structure

If upgrading from the old structure where `.env` was in the blueprint directory:

```bash
# Move existing config
mv iaac/blueprint/.env iaac/config/blueprint.env

# Update any scripts that reference the old location
# Old: source iaac/blueprint/.env
# New: source iaac/config/blueprint.env
```

## Advanced Configuration

### Dynamic Configuration

Load configuration based on environment:

```bash
# Set environment
export IAAC_ENV=production

# Use environment-specific config
iaac --env-file config/${IAAC_ENV}.env cluster up
```

### Configuration Validation

The tool validates configuration on load:

```bash
# Missing required variables will error
iaac cluster up
# Error: required configuration 'DB_PASSWORD' not set

# Check all required variables
iaac config validate
```

### Configuration Export

Export current configuration:

```bash
# Export as shell script
iaac config export --format=shell > myconfig.sh

# Export as JSON
iaac config export --format=json > myconfig.json
```

## Related Documentation

- [Environment Variables Reference](./ENV_VARS.md)
- [Agent Configuration](./AGENT.md)
- [Deployment Guide](./DEPLOYMENT.md)