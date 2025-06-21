# Configuration Directory

This directory contains all configuration files for the iaac (Infrastructure as Code) tool.

## Structure

```
config/
├── README.md                    # This file
├── blueprint.env               # Blueprint environment configuration
├── blueprint.env.example       # Example blueprint configuration
├── agent.yaml.example          # Example NLP agent configuration
├── deploy-config.yaml          # Deployment configuration
└── environments/               # Environment-specific configs (future)
    ├── local/
    ├── dev/
    └── prod/
```

## Configuration Files

### blueprint.env
Contains environment variables for:
- Cluster configuration (K3D_CLUSTER_NAME, K3D_K3S_VERSION, etc.)
- Application settings (APP_VERSION, APP_PORT, etc.)
- Database configuration (DB_NAME, DB_PASSWORD, etc.)
- Redis settings (REDIS_PASSWORD, etc.)
- Monitoring configuration

### agent.yaml
Configuration for the NLP agent including:
- OpenAI API settings
- Command whitelists/blacklists
- Safety settings
- Audit logging configuration

### deploy-config.yaml
Deployment configuration for the application including:
- Resource limits and requests
- Scaling parameters
- Health check settings

## Usage

### Using Config Directory

1. **Default location** (./config):
   ```bash
   iaac cluster up
   ```

2. **Custom config directory**:
   ```bash
   iaac --config-dir /path/to/config cluster up
   ```

3. **Specific config file**:
   ```bash
   iaac --env-file /path/to/custom.env cluster up
   ```

### Environment Variables

The tool looks for configuration in this order:
1. Command-line flags
2. Environment variables
3. Config files in the config directory
4. Default values

### Examples

```bash
# Use custom config directory
iaac --config-dir ~/myproject/config cluster up

# Use specific environment file
iaac --env-file ~/myproject/prod.env deploy

# Override specific values
CLUSTER_NAME=custom iaac cluster up
```

## Best Practices

1. **Never commit sensitive data**: Use `.env.example` files as templates
2. **Environment-specific configs**: Use separate files for different environments
3. **Version control**: Track example files, not actual config files with secrets
4. **Validation**: Always validate config before deployment

## Config File Formats

### Environment Files (.env)
```bash
# Comment
KEY=value
MULTILINE="line1
line2"
```

### YAML Files
```yaml
# Comment
key: value
nested:
  key: value
list:
  - item1
  - item2
```

## Security

- Add `*.env` to `.gitignore` (except .env.example files)
- Use strong passwords and rotate regularly
- Encrypt sensitive configuration in production
- Use environment-specific secrets management