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
- `cluster.sh`: Helper script to manage the k3d cluster and deploy resources.

## Usage

1. Make `cluster.sh` executable if needed:

   ```bash
   chmod +x cluster.sh
   ```

2. Start the cluster and deploy all components:

   ```bash
   ./cluster.sh up
   ```

   This will:
   - Create a k3d cluster named `sematic-cache`
   - Build the local Docker image `sematic-cache:latest` (using `deploy/docker/Dockerfile`) and import it into the cluster
   - Expose port `8080` on your localhost to the Sematic Cache service
   - Apply all Kubernetes manifests
   - Wait for deployments to be ready

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

5. Tear down the cluster:

   ```bash
   ./cluster.sh down
   ```

## Exposing Application

After running `./cluster.sh up`, the Sematic Cache API will be accessible at:

```
http://localhost:8080
```

## Cleanup

To remove the cluster and all resources:

```bash
./cluster.sh down
```
