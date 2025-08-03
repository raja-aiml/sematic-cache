# Semantic Cache CLI Tool

A comprehensive CLI tool for managing semantic cache with PostgreSQL/pgvector backend.

## Features

- **Cache Operations**: Get, set, and clear cache entries
- **Similarity Search**: Find semantically similar entries using vector embeddings
- **Database Management**: Ping, migrate, and monitor database status
- **Health Checks**: Monitor service health and readiness
- **Statistics**: View detailed cache and database statistics
- **OpenTelemetry Integration**: Full observability with traces and metrics
- **Configuration Management**: Flexible configuration via Viper (supports .env files)

## Installation

```bash
cd tools
go build -o bin/semantic-cache-cli main.go
```

## Configuration

The CLI tool uses a layered configuration approach:

1. **Default values** (built into the application)
2. **.env.app file** (non-secret application configuration)
3. **.env file** (secrets and environment-specific settings)
4. **Environment variables** (highest priority)

### Required Configuration

- `DATABASE_URL`: PostgreSQL connection string
- `OPENAI_API_KEY`: OpenAI API key for generating embeddings

### Optional Configuration

See `.env.app` for all available configuration options.

## Usage

### Database Commands

```bash
# Test database connection
./semantic-cache-cli database ping

# Run migrations (create tables and indexes)
./semantic-cache-cli database migrate

# Show database status
./semantic-cache-cli database status
```

### Cache Commands

```bash
# Get a value from cache
./semantic-cache-cli cache get "user:123"

# Set a value in cache
./semantic-cache-cli cache set "user:123" "John Doe"

# Clear cache entries
./semantic-cache-cli cache clear
./semantic-cache-cli cache clear --all  # Clear all entries
```

### Search Commands

```bash
# Search for similar entries
./semantic-cache-cli search "What is the weather like?"

# Search with custom threshold and limit
./semantic-cache-cli search "user query" --threshold 0.9 --limit 5
```

### Health Check

```bash
# Check service health
./semantic-cache-cli health

# Check specific endpoint
./semantic-cache-cli health --endpoint http://localhost:8080
```

### Statistics

```bash
# Display cache statistics
./semantic-cache-cli stats
```

## Architecture

The CLI tool follows these architectural principles:

- **KISS**: Simple and readable code
- **DRY**: Reusable components and utilities
- **SOLID**: Well-defined interfaces and separation of concerns
- **SDK-first**: Uses official Go SDKs instead of CLI commands
- **Observability**: Full OpenTelemetry integration

## Testing

```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run specific test
go test ./config -v
```

## Dependencies

- **Cobra**: CLI framework
- **Viper**: Configuration management
- **PostgreSQL with pgvector**: Vector database
- **OpenAI Go SDK**: Embedding generation
- **OpenTelemetry**: Observability
- **Zap**: Structured logging
- **GORM**: Database ORM
- **Testify**: Testing assertions

## Project Structure

```
tools/
├── main.go                 # Entry point
├── config/                 # Configuration management
│   ├── config.go
│   └── config_test.go
├── cmd/                    # Cobra commands
│   ├── root.go            # Root command
│   ├── database.go        # Database commands
│   ├── health.go          # Health check command
│   ├── search.go          # Similarity search command
│   └── stats.go           # Statistics command
├── observability/          # OpenTelemetry integration
│   └── telemetry.go
└── README.md              # This file
```