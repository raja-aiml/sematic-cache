# Docker Compose Deployment

This directory contains the Docker Compose setup to build and run the semantic cache system with all its dependencies, matching the functionality of the Kubernetes deployment.

## Prerequisites

- Docker (v20.10+)
- Docker Compose V2 (integrated into the `docker` CLI)
- Optional: set `COMPOSE_BAKE=true` to enable faster builds via BuildKit and Docker Bake

## Services

- **postgres**: PostgreSQL with pgvector extension
  - Image: `pgvector/pgvector:pg17`
  - Database: `cache` with user `cache:cache`
  - Automatically initializes vector extension
- **redis**: Redis 8.0.2 for caching
- **nginx**: Static web content server
- **app**: Go semantic cache server
  - Built from the project root using multi-stage Dockerfile
  - Supports `.env` file configuration
- **proxy**: Nginx reverse proxy for path-based routing
  - Routes `/web/*` to nginx service (static content)
  - Routes `/semantic-cache/*` to app service (API)

## Configuration

### Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```bash
# Database Configuration
DATABASE_URL=postgres://cache:cache@postgres:5432/cache?sslmode=disable

# OpenAI API Configuration
OPENAI_API_KEY=your-actual-api-key-here

# Optional: Jaeger Tracing
JAEGER_ENDPOINT=http://localhost:14268/api/traces
```

**Note**: The `.env` file is automatically loaded and should not be committed to git.

## Usage

### Quick Start

```bash
# Start all services
deploy/docker/dev.sh up

# Check status
deploy/docker/dev.sh status

# Test the deployment
deploy/docker/dev.sh test

# Stop services
deploy/docker/dev.sh down
```

### Manual Commands

```bash
# Build and start all services
COMPOSE_BAKE=true docker compose -f deploy/docker/docker-compose.yml up --build -d

# Stop and remove containers
docker compose -f deploy/docker/docker-compose.yml down

# View logs
docker compose -f deploy/docker/docker-compose.yml logs -f
```

### Development Script

The `dev.sh` script provides consistent functionality with the Kubernetes deployment:

```bash
Usage: dev.sh <command>

Commands:
  up         Build and start all services
  down       Stop and remove all services
  build      Build application image
  logs       Show logs from all services
  status     Show status of all services
  test       Test the deployment endpoints
  clean      Remove all containers, networks, and volumes
```

## Access URLs

After deployment, the services are available at:

- **Web Interface**: http://localhost:8080/web/
- **API Health**: http://localhost:8080/semantic-cache/health
- **API Metrics**: http://localhost:8080/semantic-cache/metrics
- **API Endpoints**: http://localhost:8080/semantic-cache/*

## API Testing

```bash
# Check health
curl http://localhost:8080/semantic-cache/health

# Get metrics
curl http://localhost:8080/semantic-cache/metrics

# Set cache entry
curl -X POST http://localhost:8080/semantic-cache/set \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AI?", "answer": "Artificial Intelligence", "modelName": "gpt-3.5-turbo"}'

# Get cache entry
curl -X POST http://localhost:8080/semantic-cache/get \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AI?"}'
```

## Architecture

The Docker deployment mirrors the Kubernetes setup:

1. **Path-based routing** via nginx proxy
2. **Environment-based configuration** with `.env` support
3. **Same service architecture** as K8s deployment
4. **Consistent API endpoints** and functionality
5. **Proper networking** with dedicated Docker network

## Troubleshooting

### View Logs
```bash
deploy/docker/dev.sh logs
```

### Check Service Status
```bash
deploy/docker/dev.sh status
docker compose -f deploy/docker/docker-compose.yml ps
```

### Restart Services
```bash
deploy/docker/dev.sh down
deploy/docker/dev.sh up
```

### Clean Up Everything
```bash
deploy/docker/dev.sh clean
```