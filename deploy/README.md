# Deployment Examples

This folder contains examples for running the semantic cache with Docker Compose
or on Kubernetes. Both setups start Postgres with the pgvector extension and a
Redis instance required by the cache server.

## Docker Compose

```bash
cd docker
docker-compose up -d
```

The compose file builds the cache server image, starts Postgres with pgvector,
and Redis. The server listens on `localhost:8080`.

## Kubernetes

Apply the manifests in the `k8s` directory:

```bash
kubectl apply -f k8s/pg-init-configmap.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/sematic-cache.yaml
```

Build and push the `sematic-cache` image referenced in `sematic-cache.yaml` to a
registry accessible by your cluster before applying the deployment.
