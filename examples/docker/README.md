# Docker Quick Start

This example shows how to start the caching server together with Postgres and Redis using the Docker Compose setup provided under `deploy/docker`.

## Prerequisites
- Docker 20.10 or newer
- Docker Compose V2 (included in the `docker` CLI)

## Usage
1. Copy your OpenAI API key into the `OPENAI_API_KEY` environment variable.
2. From the project root run:
```bash
docker compose -f deploy/docker/docker-compose.yml up --build
```
3. The server listens on `localhost:8080`.
4. Stop and remove the containers with:
```bash
docker compose -f deploy/docker/docker-compose.yml down
```
The compose file builds the server image from the repository and starts both databases with default credentials. It is shared with the main deployment configuration to avoid duplication.
