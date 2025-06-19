# Composite Backend Demo

This example demonstrates the three-tier composite backend architecture combining Memory, Redis, and PostgreSQL.

## Prerequisites

1. Redis running on localhost:6379
2. PostgreSQL with pgvector extension
3. OpenAI API key

## Setup

```bash
# Start Redis
docker run -d -p 6379:6379 redis:alpine

# Start PostgreSQL with pgvector
docker run -d -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  ankane/pgvector

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export DATABASE_URL="postgres://postgres:postgres@localhost/postgres?sslmode=disable"

# Initialize database
psql $DATABASE_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Running the Demo

```bash
cd examples/composite-demo
go run main.go -config ../../config/composite-example.yml
```

## What It Demonstrates

1. **Multi-tier caching**: Data is stored across all three tiers
2. **Automatic fallback**: Cache misses in higher tiers fall back to lower tiers
3. **Cache promotion**: Frequently accessed data is promoted to faster tiers
4. **Similarity search**: PostgreSQL tier provides vector similarity search
5. **Per-tier metrics**: Track performance of each tier independently

## Performance Characteristics

- **Memory (L1)**: <1μs latency, limited capacity
- **Redis (L2)**: ~100μs latency, distributed, no vector search
- **PostgreSQL (L3)**: 1-10ms latency, unlimited capacity, vector search

## Configuration Options

See `config/composite-example.yml` for all available options:
- Tier priorities and capacities
- Eviction policies per tier
- Promotion strategies
- Similarity thresholds