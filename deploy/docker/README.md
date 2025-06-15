# Docker Compose Deployment

This directory contains the Docker Compose setup to build and run the caching server along with its dependencies.

## Prerequisites

- Docker (v20.10+)
- Docker Compose V2 (integrated into the `docker` CLI)
- Optional: set `COMPOSE_BAKE=true` to enable faster builds via BuildKit and Docker Bake

## Services

- **postgres**: PostgreSQL with pgvector extension
  - Image: `pgvector/pgvector:pg17`
  - Environment:
    - `POSTGRES_DB=cache`
    - `POSTGRES_USER=cache`
    - `POSTGRES_PASSWORD=cache`
- **redis**: Redis 7
- **app**: Go caching server
  - Built from the project root (`../../`) using `deploy/docker/Dockerfile`
  - Environment:
    - `DATABASE_URL` (e.g., `postgres://cache:cache@postgres:5432/cache?sslmode=disable`)
    - `OPENAI_API_KEY`

## Usage

Build and start all services:
```shell
COMPOSE_BAKE=true docker compose -f deploy/docker/docker-compose.yml up --build
```

This will:
1. Build the Go server binary in a multi-stage Dockerfile.
2. Start `postgres`, `redis`, and the `app` server (listening on `:8080`).

Stop and remove the containers:
```shell
docker compose -f deploy/docker/docker-compose.yml down
```

## Customization

- To change server settings, edit environment variables in `docker-compose.yml` or pass flags to the binary.
- The multi-stage build produces a minimal final image (`debian:stable-slim`) containing only `/app/server`.