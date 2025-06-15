# Example Programs

This folder contains small programs and setup instructions that help you get started with the semantic cache.

## Prerequisites
- Go 1.23 or newer
- Docker 20.10+ with Compose V2 (for the Docker example)

## Usage
Each subdirectory includes a README with details:

### simple
Minimal in-memory cache example.
```bash
go run ./examples/simple
```

### advanced
Demonstrates inner-product similarity and a naive ANN index.
```bash
go run ./examples/advanced
```

### docker
Quick start for running the server with PostgreSQL and Redis.
```bash
docker compose -f deploy/docker/docker-compose.yml up --build
```
Stop the containers with:
```bash
docker compose -f deploy/docker/docker-compose.yml down
```
See `docker/README.md` for more information.
