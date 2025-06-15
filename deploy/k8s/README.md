# Kubernetes Deployment for Sematic Cache

This directory contains Kubernetes manifests and helper scripts to deploy the Sematic Cache application using a local [k3d](https://k3d.io/) cluster.

## Prerequisites

- Docker
- [k3d](https://k3d.io/) (tested with v5+)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)

## Files

- `pg-init-configmap.yaml`: ConfigMap to initialize PostgreSQL with the `vector` extension.
- `postgres.yaml`: PVC, Deployment, and Service for PostgreSQL.
- `redis.yaml`: Deployment and Service for Redis.
- `sematic-cache.yaml`: Deployment and Service for the Sematic Cache API.
- `registry.yaml`: Deployment and Service for a local Docker registry.
- `cluster.sh`: Manage the k3d cluster (create, delete, logs).
- `dev.sh`: CLI to build the image, push to the local registry, and deploy.

## Usage

1. Make the scripts executable if needed:

   ```bash
   chmod +x cluster.sh dev.sh
   ```

2. Build the image and deploy everything:

   ```bash
   ./dev.sh deploy
   ```

   This will:
   - Start a local Docker registry `sematic-registry` on port `5000`
   - Build the Docker image `sematic-cache:latest` and push it to the registry
   - Create a k3d cluster named `sematic-cache`
   - Apply all Kubernetes manifests and wait for deployments

3. View the status of the cluster:

   ```bash
   ./cluster.sh ps
   ```

4. View logs of all pods:

   ```bash
   ./cluster.sh logs
   ```

   To follow logs of a specific pod:

   ```bash
   ./cluster.sh logs <pod-name> --follow
   ```

5. Tear down the cluster and registry:

   ```bash
   ./dev.sh down
   ```
   This command removes the k3d cluster and stops the `sematic-registry` container.

## Exposing Application

After running `./dev.sh deploy`, the Sematic Cache API will be accessible at:

```
http://localhost:8080
```

## Cleanup

To remove the cluster, registry, and all resources:

```bash
./dev.sh down
```
