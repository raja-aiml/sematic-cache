# Environment Variables Configuration

The iaac tool supports extensive configuration through environment variables, making it adaptable to different applications and environments.

## Application Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `IAAC_APP_NAME` | Application name | `myapp` |
| `IAAC_SECRET_NAME` | Kubernetes secret name for app | `app-secrets` |
| `IAAC_BLUEPRINT_PATH` | Path to blueprint directory | `iaac/blueprint` |
| `IAAC_DATABASE_NAME` | Database name | `appdb` |

## Cluster Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `IAAC_CLUSTER_NAME` | k3d cluster name | `local-k8s` |
| `IAAC_API_PORT` | Kubernetes API port | `6550` |
| `IAAC_HTTP_PORT` | HTTP port mapping | `8080:80` |
| `IAAC_HTTPS_PORT` | HTTPS port mapping | `8443:443` |

## Build Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `IAAC_IMAGE_NAME` | Docker image name | `app:local` |
| `IAAC_DOCKERFILE` | Dockerfile path | `Dockerfile` |
| `IAAC_BUILD_CONTEXT` | Docker build context | `.` |

## Application-Specific Secrets

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI) | No |
| `DATABASE_URL` | Database connection string | No |

## Debug and Development

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug logging | `false` |
| `SC_DEPLOY_DEBUG` | Enable deployment debug mode | `false` |

## Usage Examples

### Basic Usage (defaults)
```bash
iaac cluster up
```

### Custom Application
```bash
export IAAC_APP_NAME=myservice
export IAAC_CLUSTER_NAME=myservice-dev
export IAAC_IMAGE_NAME=myservice:latest
export IAAC_DATABASE_NAME=myservice_db
export IAAC_SECRET_NAME=myservice-secrets

iaac workflow full
```

### Different Blueprint Path
```bash
export IAAC_BLUEPRINT_PATH=deploy/k8s
iaac cluster up
```

### Custom Ports
```bash
export IAAC_HTTP_PORT=3000:80
export IAAC_HTTPS_PORT=3443:443
iaac cluster up
```

## Configuration File Override

You can also create a `deploy-config.yaml` file to override defaults:

```yaml
app:
  name: myservice
  secret_name: myservice-secrets
  blueprint_path: deploy/k8s
  database_name: myservice_db

cluster:
  name: myservice-local
  http_port: "3000:80"
  https_port: "3443:443"

build:
  image_name: myservice:local
```

The tool will look for config files in:
1. `./deploy-config.yaml`
2. `./config/deploy-config.yaml`
3. `~/.iaac/deploy-config.yaml`

Environment variables take precedence over config file values.