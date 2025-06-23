# Deploy Tasks

This directory contains deployment-related task definitions organized by platform/tool.

## Available Task Files

### k3d.yaml
k3d local cluster management:
- **k3d:create** - Create local k3d cluster
- **k3d:delete** - Delete cluster
- **k3d:start/stop** - Start/stop cluster
- **k3d:registry:** - Local registry management
- **k3d:image:** - Image import operations

### k8s.yaml
Kubernetes deployment operations:
- **k8s:deploy** - Deploy application
- **k8s:rollback** - Rollback deployment
- **k8s:scale** - Scale deployment
- **k8s:logs** - View logs
- **k8s:exec** - Execute commands in pods
- **k8s:port-forward** - Port forwarding

### helm.yaml
Helm chart management:
- **helm:install** - Install chart
- **helm:upgrade** - Upgrade release
- **helm:rollback** - Rollback release
- **helm:uninstall** - Remove release
- **helm:lint** - Validate chart
- **helm:test** - Run chart tests

## Usage

Include the task files you need in your Taskfile.yaml:

```yaml
includes:
  k3d: ./devops/tasks/deploy/k3d.yaml
  k8s: ./devops/tasks/deploy/k8s.yaml
  helm: ./devops/tasks/deploy/helm.yaml

tasks:
  setup:
    desc: Setup local environment
    cmds:
      - task: k3d:create
      - task: k8s:namespace:create
        vars: {NAMESPACE: app}
      
  deploy:
    desc: Deploy application
    cmds:
      - task: k8s:deploy
        vars: {NAMESPACE: app}
```

## Variables

### k3d Tasks
- `CLUSTER_NAME` - Cluster name (default: dev-cluster)
- `K3S_VERSION` - k3s version (default: v1.28.3-k3s1)
- `API_PORT` - API server port (default: 6443)
- `LB_PORT` - Load balancer port (default: 8080)

### k8s Tasks
- `NAMESPACE` - Kubernetes namespace (default: default)
- `APP_NAME` - Application name (default: app)
- `MANIFEST_DIR` - Manifest directory (default: ./k8s)
- `DEPLOY_TIMEOUT` - Deployment timeout (default: 5m)

### Helm Tasks
- `RELEASE_NAME` - Helm release name
- `CHART_PATH` - Chart directory (default: ./chart)
- `VALUES_FILE` - Values file (default: values.yaml)
- `HELM_TIMEOUT` - Operation timeout (default: 5m)

## Best Practices

1. **Use namespaces** - Always specify namespace for isolation
2. **Set timeouts** - Configure appropriate timeouts for operations
3. **Validate first** - Run dry-run/validate before actual deployment
4. **Check status** - Always verify deployment status after operations
5. **Use labels** - Consistent labeling for resource selection