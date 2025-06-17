# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Build and Run
```bash
# Build the server binary
go build -o bin/server cmd/server/main.go

# Run the server with config file
go run cmd/server/main.go -config config.yml

# Run with environment variables only
go run cmd/server/main.go -address :8080
```

### Testing
```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run specific package tests
go test ./core
go test ./storage
go test ./server
```

### Kubernetes Development
```bash
# Create k3d cluster and deploy infrastructure
deploy/cluster.sh up

# Build and deploy application
deploy/dev.sh build
deploy/dev.sh deploy

# Check status and logs
deploy/cluster.sh ps
deploy/cluster.sh logs

# Cleanup
deploy/cluster.sh down
```

## Architecture

### Core Components

**Cache System (`core/`)**
- `cache.go`: Main cache implementation with embedding-based similarity search
- `agent.go`: AI agent system with context chain management
- `orchestrator.go`: Multi-agent orchestration with intelligent routing
- Supports multiple eviction policies (LRU, FIFO, LFU, Random)
- TTL support with automatic expiration
- Adaptive thresholding for dynamic similarity matching

**Storage Backends (`storage/`)**
- `pgstore.go`: PostgreSQL with pgvector for persistent vector similarity search
- `redisstore.go`: Redis cluster support for distributed caching
- `gormstore.go`: GORM integration for advanced PostgreSQL operations
- In-memory cache with configurable capacity

**Server (`server/`)**
- `server.go`: HTTP REST API with Gin framework
- `auth.go`: Authentication middleware
- Endpoints: `/set`, `/get`, `/query`, `/topk`, `/health`, `/metrics`

**OpenAI Integration (`openai/`)**
- `client.go`: Complete OpenAI API wrapper
- Support for chat completions, embeddings, image generation, audio processing
- Streaming support for chat completions

**Configuration (`config/`)**
- YAML-based configuration system
- Environment variable support
- Database, Redis, and OpenAI API configuration

### Key Patterns

**Embedding-Based Similarity**
- Uses cosine similarity, inner product, or L2 distance
- Automatic embedding generation via OpenAI API
- Configurable similarity thresholds (default: 0.8)

**Agent System**
- Context chains with automatic message eviction
- Expert agent roles with specialized prompts
- Multi-agent orchestration with keyword-based routing
- Integration with semantic caching for performance

**Storage Abstraction**
- Common interface across memory, Redis, and PostgreSQL backends
- Automatic TTL handling and expiration
- Batch operations for efficient bulk loading

## Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
export DATABASE_URL="postgres://user:pass@localhost/dbname?sslmode=disable"
export JAEGER_ENDPOINT="http://localhost:14268/api/traces"
```

## Configuration

Example `config.yml`:
```yaml
server:
  address: ":8080"
cache:
  type: "gorm"  # Options: memory, redis, gorm
  capacity: 1000
  eviction_policy: "LRU"
  ttl: "1h"
  min_similarity: 0.8
openai:
  api_key: "${OPENAI_API_KEY}"
  base_url: "https://api.openai.com/v1"
database_url: "host=localhost user=postgres dbname=cache sslmode=disable"
```

## Development Notes

- PostgreSQL requires the `vector` extension for pgvector functionality
- K8s deployment uses k3d for local development with no registry complexity
- Kubernetes deployment includes automatic database initialization with extensions
- All test files follow Go testing conventions with descriptive test names
- The codebase uses OpenTelemetry for observability with Jaeger integration
- Single deployment target (Kubernetes) eliminates Docker/K8s maintenance overhead

## Commit Message Guidelines

- Use conventional commits format (feat:, fix:, refactor:, etc.)
- Do NOT include "🤖 Generated with Claude Code" or similar AI attribution in commit messages
- Keep commit messages concise but descriptive
- Focus on the "why" and "what" of changes
- Use present tense ("add feature" not "added feature")

## Code Generation Attribution

This repository contains code and documentation generated with assistance from Claude Code (claude.ai/code). The AI assistance includes:
- Code refactoring and organization
- Documentation writing and updates
- Deployment script creation
- Configuration file generation
- README and guide creation

All generated content has been reviewed and integrated as part of the normal development process.