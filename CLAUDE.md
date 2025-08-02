# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with the semantic cache repository. It emphasizes architectural principles, code quality standards, and development best practices.

## 🚨 STRICT ARCHITECTURE RULES - MUST FOLLOW 🚨

### 1. ALWAYS USE SDK OVER CLI COMMANDS
- **MANDATORY**: Use official Go SDKs/libraries instead of CLI commands
- **FORBIDDEN**: Running external commands via exec/shell unless absolutely necessary
- **EXCEPTION**: Only when no SDK exists and must be documented
- **Example**: Use Docker SDK instead of `docker` CLI, OpenAI SDK instead of HTTP calls

### 2. APPROVED TECHNOLOGY STACK
**Web Frameworks:**
- ✅ **Gin** (github.com/gin-gonic/gin) - Primary web framework
- ❌ Fiber, Echo, Chi - Not approved

**Database/Storage:**
- ✅ **PostgreSQL** with pgvector extension - Vector similarity search
- ✅ **GORM** for ORM with pgvector-go driver
- ✅ **pgx** driver for direct PostgreSQL access
- ❌ MongoDB, MySQL, Redis - Not implemented

**Testing:**
- ✅ **testify** (github.com/stretchr/testify) - Assertions and mocks
- ✅ **Table-driven tests** - Required pattern
- ✅ **httptest** - For HTTP testing
- ❌ Ginkgo, Gomega - Not approved

**Configuration:**
- ✅ **YAML configuration** - Simple config file loading
- ✅ **Environment variables** - Twelve-Factor App methodology
- ✅ **Structured config types** - Type-safe configuration
- ✅ **Viper** (github.com/spf13/viper) - Advanced configuration management
- ✅ **Cobra** (github.com/spf13/cobra) - CLI framework for command-line tools

**Observability:**
- ✅ **OpenTelemetry** - Complete observability with OTLP exporters
- ✅ **Zap** (go.uber.org/zap) - Structured logging with OTel integration
- ✅ **Prometheus** - Metrics collection
- ✅ **Jaeger** - Distributed tracing
- ✅ **OpenTelemetry Collector** - Telemetry pipeline


**Container/Development:**
- ✅ **Docker** - Containerization with multi-stage builds
- ✅ **Docker Compose** - Local development environment
- ✅ **Air** - Hot reload for development


### 3. INTERFACE COMPLIANCE
- **MANDATORY**: Always check existing interfaces before implementation
- **MANDATORY**: Run tests to verify interface compliance
- **FORBIDDEN**: Creating duplicate functionality
- **RULE**: If an interface exists, implement it correctly or propose changes

### 4. ERROR HANDLING
```go
// CORRECT
if err != nil {
    return fmt.Errorf("failed to do X: %w", err)
}

// INCORRECT
if err != nil {
    return err // Missing context
}
```

### 5. DEPENDENCY MANAGEMENT
- **MANDATORY**: Run `go mod tidy` after adding dependencies
- **MANDATORY**: Check for duplicate functionality before adding new deps
- **MANDATORY**: Verify compatibility with Go 1.23.x
- **FORBIDDEN**: Adding dependencies that duplicate existing functionality

## Quick Start Summary

### Essential Commands
```bash
gofmt -w .              # Format all code (MANDATORY before commit)
go test ./...           # Run all tests (MANDATORY before commit)
go vet ./...            # Static analysis (MANDATORY before commit)
go mod tidy             # Clean dependencies (MANDATORY after changes)
go run main.go              # Run server
make dev                    # Run with hot reload
make test                   # Run all tests
make lint                   # Run linter
```

### Git Commit Guidelines
When you need to quickly stage all changes and commit with an auto-generated message, use this instruction:
```
"git add, generate git commit message and commit"
```
This will:
1. Check git status to understand changes
2. Stage all modified and new files
3. Generate a descriptive commit message following conventional commits format
4. Create the commit without any AI attribution

**IMPORTANT: Never include these lines in commit messages:**
```
🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```


### Key Principles to Follow
1. **KISS**: Keep implementations simple and readable
2. **DRY**: Don't repeat code - extract common functionality
3. **SOLID**: Follow all five SOLID principles
4. **Testing**: Minimum 80% coverage with table-driven tests
5. **Error Handling**: Never ignore errors, always wrap with context
6. **Interfaces**: Depend on abstractions, not concrete types
7. **Context**: Always pass context.Context as first parameter
8. **Formatting**: Code MUST pass gofmt before committing
9. **SDK First**: ALWAYS prefer SDK/library over CLI commands

### Project Structure
- `internal/`: Internal packages (following Go best practices)
  - `cache/`: Cache handlers and API types
  - `config/`: Configuration management (YAML and env-based)
  - `database/`: Database connection management
  - `embedding/`: OpenAI client and dimension reduction algorithms
  - `logger/`: Zap-based structured logging with OTel integration
  - `observability/`: OpenTelemetry setup for traces and metrics
  - `server/`: Gin HTTP server and routing
  - `storage/`: Storage backend implementations with adapter pattern
- `cmd/`: Application orchestration and entry point
- `deployments/`: Deployment configurations
  - `docker/`: Dockerfile for containerization
  - `local/`: Docker Compose for local development with full observability stack
- `bin/`: Build output directory (gitignored)

## Architectural Principles

### KISS (Keep It Simple, Stupid)
- Prefer simple, readable solutions over clever ones
- Each function should do one thing well
- Avoid premature optimization
- Clear naming over comments
- Example: Use standard library when possible instead of external dependencies

### DRY (Don't Repeat Yourself)
- Extract common functionality into reusable functions
- Use interfaces to share behavior across types
- Centralize configuration and constants
- Create shared utilities for common operations
- Example: Single embedding generation function used by all storage backends

### SOLID Principles

**Single Responsibility Principle (SRP)**
- Each struct/type should have one reason to change
- Separate concerns into different packages
- Example: Storage backends handle persistence, not similarity calculations

**Open/Closed Principle (OCP)**
- Open for extension, closed for modification
- Use interfaces and composition over inheritance
- Example: Storage interface allows adding new backends without modifying core

**Liskov Substitution Principle (LSP)**
- Subtypes must be substitutable for their base types
- All storage backends must fully implement the Storage interface
- Example: Storage interface allows different backend implementations

**Interface Segregation Principle (ISP)**
- Clients should not depend on interfaces they don't use
- Keep interfaces small and focused
- Example: Separate interfaces for basic cache operations vs. advanced queries

**Dependency Inversion Principle (DIP)**
- Depend on abstractions, not concretions
- High-level modules should not depend on low-level modules
- Example: Cache depends on Storage interface, not specific implementations

## Build Output Guidelines
- Always create build output to `bin` directory that is excluded in `.gitignore`
- Use consistent output directory for compiled binaries
- Ensure build artifacts are not committed to version control

## Code Style Requirements

### Mandatory Go Formatting
**All Go source files MUST conform to `gofmt` output.** Before committing any code:

```bash
# Format all Go code (MANDATORY)
gofmt -w .

# Run go vet (MANDATORY)
go vet ./...

# Run all tests (MANDATORY)
go test ./...

# Clean up dependencies (MANDATORY after adding/removing deps)
go mod tidy
```

### Import Organization
Imports must be organized in the following groups, separated by blank lines:
1. Standard library imports
2. Third-party imports
3. Local application imports

```go
import (
    "context"
    "fmt"
    
    "github.com/gin-gonic/gin"
    "github.com/stretchr/testify/assert"
    
    "github.com/raja-aiml/sematic-cache/internal/cache"
    "github.com/raja-aiml/sematic-cache/internal/storage"
)
```

### Error Handling
```go
// Always wrap errors with context
if err != nil {
    return fmt.Errorf("failed to connect to database: %w", err)
}

// Check errors immediately
resp, err := client.Do(req)
if err != nil {
    return nil, fmt.Errorf("request failed: %w", err)
}
defer resp.Body.Close()
```

### Testing Requirements

1. **Target 80% test coverage** for new code
2. **Table-driven tests** for multiple test cases
3. **Mock external dependencies** using interfaces
4. **Test file naming**: `*_test.go` in the same package
5. **Benchmark tests** for performance-critical code (especially embedding reduction)
6. **Extended tests** for mathematical algorithms (`*_extended_test.go`)
7. **Production tests** for real-world scenarios

**Current test coverage gaps to address:**
- Cache handlers need unit tests
- Storage implementations need integration tests
- Server components need HTTP tests
- Configuration packages need validation tests

Example table-driven test:
```go
func TestCache(t *testing.T) {
    tests := []struct {
        name     string
        input    string
        expected string
        wantErr  bool
    }{
        {"valid input", "test", "TEST", false},
        {"empty input", "", "", true},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got, err := process(tt.input)
            if tt.wantErr {
                assert.Error(t, err)
                return
            }
            assert.NoError(t, err)
            assert.Equal(t, tt.expected, got)
        })
    }
}
```

### Interface Design
- Keep interfaces small and focused
- Define interfaces in the package that uses them
- Use interface composition for complex behaviors

```go
type Reader interface {
    Read(ctx context.Context, key string) (string, error)
}

type Writer interface {
    Write(ctx context.Context, key, value string) error
}

type ReadWriter interface {
    Reader
    Writer
}
```

## Performance Guidelines

1. **Profile before optimizing** - Use pprof and benchmarks
2. **Avoid premature optimization** - Write clear code first
3. **Use sync.Pool** for frequently allocated objects
4. **Minimize allocations** in hot paths
5. **Use buffered channels** when appropriate

## Security Requirements

1. **Never log sensitive data** (passwords, tokens, PII)
2. **Use context for cancellation** and timeouts
3. **Validate all inputs** especially from external sources
4. **Use prepared statements** for SQL queries
5. **Follow principle of least privilege** for permissions

## Code Review Checklist

Before submitting code, ensure:
- [ ] Code passes `gofmt -w .`
- [ ] Code passes `go vet ./...`
- [ ] All tests pass `go test ./...`
- [ ] Test coverage for new code >= 80%
- [ ] No CLI commands used where SDK exists
- [ ] Interfaces properly implemented
- [ ] Errors wrapped with context
- [ ] Dependencies cleaned with `go mod tidy`
- [ ] Documentation updated if needed
- [ ] No sensitive data in logs or commits
- [ ] Observability added (traces/metrics/logs)
- [ ] Docker image builds successfully
- [ ] Integration with existing storage adapter verified

## Common Pitfalls to Avoid

1. **Using CLI instead of SDK**: Always check for Go SDK first
2. **Ignoring interfaces**: Check existing interfaces before creating new types
3. **Poor error messages**: Always add context to errors
4. **Missing tests**: Every new feature needs tests
5. **Forgetting go mod tidy**: Always run after dependency changes
6. **Using wrong web framework**: Use Gin, not others
7. **Creating duplicate functionality**: Search codebase first
8. **Ignoring existing patterns**: Follow the adapter pattern in storage layer
9. **Skipping observability**: Always add tracing and metrics to new features
10. **Direct OpenAI API calls**: Use the internal embedding client wrapper

## Examples of Good Practices

### Using SDK instead of CLI

**Docker Example:**
```go
// BAD - Using CLI
out, err := exec.Command("docker", "build", "-t", tag, ".").Output()

// GOOD - Using SDK
client, err := docker.NewClient()
if err != nil {
    return fmt.Errorf("failed to create docker client: %w", err)
}
defer client.Close()

err = client.Build(ctx, docker.BuildOptions{
    Tags: []string{tag},
    Context: ".",
})
```

**OpenAI Example:**
```go
// BAD - Using raw HTTP calls
resp, err := http.Post("https://api.openai.com/v1/embeddings", ...)

// GOOD - Using official SDK
import "github.com/openai/openai-go/v2"

client := openai.NewClient(
    option.WithAPIKey(apiKey),
)

embedding, err := client.Embeddings.New(ctx, openai.EmbeddingNewParams{
    Input: openai.F([]string{text}),
    Model: openai.F(openai.EmbeddingModelTextEmbedding3Small),
})
if err != nil {
    return fmt.Errorf("failed to generate embedding: %w", err)
}
```

### Proper Interface Implementation
```go
// First, check the interface
type CacheBackend interface {
    Get(ctx context.Context, key string) (interface{}, error)
    Set(ctx context.Context, key string, value interface{}) error
    Delete(ctx context.Context, key string) error
    Clear(ctx context.Context) error
    GetSimilar(ctx context.Context, embedding []float32, threshold float64, limit int) ([]SimilarItem, error)
    GetStats(ctx context.Context) (CacheStats, error)
}

// Then implement correctly
type MyCache struct{}

func (c *MyCache) Get(ctx context.Context, key string) (interface{}, error) {
    // Implementation with context and error handling
    return nil, nil
}

func (c *MyCache) Set(ctx context.Context, key string, value interface{}) error {
    // Implementation with context
    return nil
}

// Verify at compile time
var _ CacheBackend = (*MyCache)(nil)
```

## Observability Stack

### OpenTelemetry Integration
The project uses a comprehensive observability stack with OpenTelemetry at its core:

```go
// Initialize observability
import "github.com/raja-aiml/sematic-cache/internal/observability"

shutdown, err := observability.SetupOtelSDK(ctx, serviceName, serviceVersion)
if err != nil {
    return fmt.Errorf("failed to setup OpenTelemetry: %w", err)
}
defer shutdown(ctx)
```

### Structured Logging with Zap
```go
import "github.com/raja-aiml/sematic-cache/internal/logger"

// Initialize logger
log := logger.NewLogger("production") // or "development"

// Use structured logging
log.Info("cache hit",
    zap.String("key", key),
    zap.Duration("latency", latency),
    zap.Float64("similarity", similarity),
)
```

### Local Development Stack
The project includes a complete observability stack via Docker Compose:
- **PostgreSQL with pgvector**: Vector database for embeddings
- **OpenTelemetry Collector**: Telemetry pipeline
- **Jaeger**: Distributed tracing UI
- **Prometheus**: Metrics collection
- **Caddy**: Reverse proxy for services

```bash
# Start the observability stack
docker-compose -f deployments/local/docker-compose.yml up -d

# Access services:
# - Application: http://localhost:8080
# - Jaeger UI: http://localhost:16686
# - Prometheus: http://localhost:9090
```

## Embedding and AI Integration

### OpenAI Client
The project uses the official OpenAI Go SDK v2:

```go
import "github.com/openai/openai-go/v2"

// Client supports:
// - Chat completions (streaming and non-streaming)
// - Text embeddings generation
// - Image generation and editing
// - Audio transcription and translation
// - Content moderation
```

### Dimension Reduction System
Advanced embedding optimization in `internal/embedding/reduction/`:
- **PCA**: Principal Component Analysis for dimension reduction
- **Incremental PCA**: Memory-efficient for large datasets
- **Adaptive algorithms**: Auto-tuning based on data characteristics
- **Performance optimization**: SIMD operations using Gonum

### Storage Backend with pgvector
```go
// Vector similarity search using cosine distance
type Store struct {
    db *gorm.DB
}

// Automatic indexing with IVFFlat for performance
// Cosine similarity search for semantic matching
```

## Development Workflow

### Hot Reload Development
```bash
# Install air for hot reload
go install github.com/air-verse/air@latest

# Run with hot reload
make dev
```

### Testing Strategy
```bash
# Run all tests
make test

# Run with coverage
go test -cover ./...

# Run benchmarks
go test -bench=. ./internal/embedding/reduction/...

# Watch mode for TDD
make test-watch
```

### API Endpoints
The Gin server exposes the following endpoints:
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `POST /api/v1/get` - Retrieve cached response
- `POST /api/v1/set` - Store new cache entry
- `POST /api/v1/similar` - Find similar entries
- `GET /api/v1/stats` - Cache statistics
- `POST /api/v1/clear` - Clear cache entries

## Current Implementation Status

### Implemented Features
- ✅ PostgreSQL with pgvector for vector similarity search
- ✅ OpenAI integration for embeddings and chat
- ✅ Dimension reduction algorithms for embedding optimization
- ✅ Complete observability stack with OpenTelemetry
- ✅ Gin-based REST API with health checks
- ✅ Docker containerization with multi-stage builds
- ✅ Structured logging with Zap
- ✅ Configuration via YAML and environment variables


### Key Dependencies
- Go 1.24.x
- PostgreSQL with pgvector extension
- OpenAI API for embeddings
- OpenTelemetry for observability
- Docker for containerization

This document is the source of truth for all development decisions in this repository.