# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with the semantic cache repository. It emphasizes architectural principles, code quality standards, and development best practices.

## Quick Start Summary

### Essential Commands
```bash
gofmt -w .              # Format all code (MANDATORY before commit)
go test ./...           # Run all tests (MANDATORY before commit)
go vet ./...            # Static analysis (MANDATORY before commit)
go run cmd/server/main.go -config config.yml  # Run server
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

### Project Structure
- `core/`: Core functionality (cache, agents, orchestrator)
- `storage/`: Storage backend implementations
- `server/`: HTTP API server
- `openai/`: OpenAI integration
- `config/`: Configuration management
- `cmd/`: Application entry points

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
- Example: Switching between Redis and PostgreSQL should require no code changes

**Interface Segregation Principle (ISP)**
- Clients should not depend on interfaces they don't use
- Keep interfaces small and focused
- Example: Separate interfaces for basic cache operations vs. advanced queries

**Dependency Inversion Principle (DIP)**
- Depend on abstractions, not concretions
- High-level modules should not depend on low-level modules
- Example: Cache depends on Storage interface, not specific implementations

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
```

### Naming Conventions
- Use `MixedCaps` or `mixedCaps` (camelCase) - **never underscores**
- Package names: all lowercase, single words when possible
- Variable names: length proportional to scope
- Receivers: 1-2 character abbreviations, consistent throughout type
- Constants: use `MixedCaps` not `SCREAMING_SNAKE_CASE`

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

# Run tests with race detection
go test -race ./...

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

### Design Philosophy

**Modularity**
- Clear separation of concerns between packages
- Well-defined interfaces between components
- Minimal coupling, high cohesion

**Testability**
- Dependency injection for all external dependencies
- Interface-based design for easy mocking
- Pure functions where possible

**Performance**
- Efficient data structures (pre-allocated slices, maps)
- Connection pooling for databases
- Concurrent operations where beneficial
- Caching at multiple levels

**Maintainability**
- Self-documenting code through clear naming
- Consistent patterns across the codebase
- Comprehensive error handling
- Structured logging for debugging

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

### Package Structure
```
cmd/                    # Application entry points
├── server/            # Main server application
└── cli/               # Command-line tools

internal/              # Private application code (if using)
├── cache/            # Core caching logic
├── storage/          # Storage implementations
├── embedding/        # OpenAI embedding client
└── similarity/       # Similarity algorithms

core/                  # Core package (current structure)
storage/              # Storage implementations
server/               # HTTP server
openai/               # OpenAI integration
config/               # Configuration management
```

### Key Patterns

**Embedding-Based Similarity**
- Uses cosine similarity, inner product, or L2 distance
- Automatic embedding generation via OpenAI API
- Configurable similarity thresholds (default: 0.8)

**Agent System Architecture**
The semantic cache includes a sophisticated multi-agent system designed with clear architectural principles:

1. **Context Chain Management** (`core/agent.go`)
   - Maintains conversation history with automatic eviction
   - Configurable context window size (default: 10 messages)
   - Preserves semantic coherence across interactions
   - Thread-safe operations with proper locking
   - Memory-efficient message storage

2. **Expert Agents** 
   - Specialized agents with domain-specific prompts
   - Role-based routing and expertise
   - Integration with semantic cache for performance
   - Each agent follows SRP - focused on one domain
   - Extensible design for adding new expert types

3. **Multi-Agent Orchestration** (`core/orchestrator.go`)
   - Intelligent routing based on query keywords
   - Parallel agent execution support
   - Result aggregation and synthesis
   - Dynamic agent selection
   - Circuit breaker pattern for resilience
   - Configurable routing strategies

4. **Agent Features**
   - Stateful conversation management
   - Semantic caching integration for faster responses
   - Configurable system prompts
   - Context-aware response generation
   - Automatic fallback to primary agent
   - Performance metrics and monitoring

5. **Agent Design Patterns**
   ```go
   // Example: Creating an agent with functional options
   agent := NewAgent(
       WithSystemPrompt("You are a helpful assistant"),
       WithMaxContextSize(20),
       WithCache(semanticCache),
       WithTimeout(30 * time.Second),
   )
   
   // Example: Orchestrator with multiple experts
   orchestrator := NewOrchestrator(
       WithExperts(map[string]*Agent{
           "coding": codingExpert,
           "data": dataExpert,
           "general": generalAgent,
       }),
       WithRoutingStrategy(KeywordBased),
   )
   ```

**Storage Abstraction**
- Common interface across memory, Redis, and PostgreSQL backends
- Automatic TTL handling and expiration
- Batch operations for efficient bulk loading
- Thread-safe operations with proper locking

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

## Development Guidelines

### Code Organization Principles

**Package Design**
- Packages should be cohesive and focused
- Avoid circular dependencies
- Export only what's necessary
- Group related functionality together

**Function Design**
- Functions should be short (typically < 50 lines)
- Single purpose per function
- Clear input/output contracts
- Minimize side effects

**Data Structure Guidelines**
- Choose the right data structure for the job
- Consider memory layout and cache locality
- Use value types for small data, pointers for large
- Implement proper validation methods

### Error Handling
- Always handle errors explicitly
- Use `error` as the last return value
- Never ignore errors with `_`
- Use `fmt.Errorf` with `%w` verb for error wrapping
- Early return pattern (avoid nested if/else)

### Import Organization
Group imports into exactly three blocks:
```go
import (
    // 1. Standard library
    "context"
    "fmt"
    "time"

    // 2. Third-party packages  
    "github.com/go-redis/redis/v8"
    "gorm.io/gorm"

    // 3. Local packages
    "github.com/your-org/semantic-cache/core"
    "github.com/your-org/semantic-cache/storage"
)
```

### Context Propagation
Always pass `context.Context` as the first parameter:
```go
func (c *Cache) Get(ctx context.Context, key string) (interface{}, error)
func (s *Storage) Store(ctx context.Context, embedding []float64, response string) error
```

### Testing Requirements
- Use table-driven tests for comprehensive coverage
- Create focused test helpers with `t.Helper()`
- Mock external dependencies with interfaces
- Maintain test coverage above 80%
- Follow AAA pattern: Arrange, Act, Assert
- Test edge cases and error conditions
- Use subtests for better organization
- Benchmark critical paths

Example test structure:
```go
func TestCache_Get(t *testing.T) {
    tests := []struct {
        name    string
        setup   func(*Cache)
        key     string
        want    interface{}
        wantErr bool
    }{
        // Test cases...
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // Test implementation
        })
    }
}
```

### Performance Best Practices
- Pre-allocate slices and maps when size is known
- Use `strings.Builder` for string concatenation
- Use pointers for large structs to avoid copying
- Implement connection pooling for databases

### OpenTelemetry Integration
- Always propagate OpenTelemetry context
- Use structured logging with `log/slog`
- Export Prometheus metrics for monitoring
- Implement distributed tracing

## Backend-Specific Guidelines

### PostgreSQL/pgvector
- Use prepared statements for all queries
- Implement proper connection pooling
- Use appropriate vector indexes (HNSW or IVFFlat)
- Handle database migrations carefully
- Vector similarity operators: `<=>` (L2), `<#>` (inner product), `<->` (cosine)

### Redis
- Handle connection failures gracefully
- Implement circuit breaker pattern
- Use Redis pipelines for batch operations
- Set appropriate TTLs for cached data
- Support for Redis Cluster configuration

### OpenAI Integration
- Handle rate limiting with exponential backoff
- Implement proper timeout handling
- Use context for request cancellation
- Never log API keys or sensitive data
- Circuit breaker pattern for resilience

## Development Notes

- PostgreSQL requires the `vector` extension for pgvector functionality
- K8s deployment uses k3d for local development with no registry complexity
- Kubernetes deployment includes automatic database initialization with extensions
- All test files follow Go testing conventions with descriptive test names
- The codebase uses OpenTelemetry for observability with Jaeger integration
- Single deployment target (Kubernetes) eliminates Docker/K8s maintenance overhead
- Use functional options pattern for complex configuration
- Implement circuit breaker pattern for external services
- All storage backends implement a common interface for consistency

## Commit Message Guidelines

- Use conventional commits format (feat:, fix:, refactor:, etc.)
- Do NOT include "🤖 Generated with Claude Code" or similar AI attribution in commit messages
- Keep commit messages concise but descriptive
- Focus on the "why" and "what" of changes
- Use present tense ("add feature" not "added feature")
- DO NOT PUT Generated with Claude Code and Co-Authored-By: Claude 

## Common Design Patterns

### Functional Options Pattern
```go
type CacheOption func(*CacheConfig)

func WithMaxSize(size int) CacheOption {
    return func(c *CacheConfig) {
        c.MaxSize = size
    }
}

cache, err := NewCache(
    WithMaxSize(5000),
    WithSimilarityThreshold(0.9),
)
```

### Interface-Based Design
- Define interfaces for all external dependencies
- Use dependency injection for testability
- Create mock implementations for testing

### Concurrency Patterns
- Make goroutine lifetimes explicit
- Use channels with specified directions
- Prefer synchronous functions for clarity
- Context propagation for cancellation

## Quick Reference

### Required Checks Before Committing
```bash
gofmt -w .              # Format code
go vet ./...            # Static analysis
go test ./...           # Run tests
go test -race ./...     # Race detection
```

### Common Interfaces
- `Storage`: Common interface for all storage backends
- `Embedder`: Interface for embedding generation
- `SimilarityCalculator`: Interface for similarity algorithms
- `Agent`: Interface for AI agents
- `ContextChain`: Interface for conversation management
- `Orchestrator`: Interface for multi-agent coordination

### Code Review Checklist

Before submitting code, ensure:
- [ ] Code follows Go formatting standards (`gofmt`)
- [ ] All tests pass (`go test ./...`)
- [ ] No race conditions (`go test -race ./...`)
- [ ] Error handling is comprehensive
- [ ] Interfaces are properly implemented
- [ ] No code duplication (DRY)
- [ ] Functions have single responsibility (SRP)
- [ ] Dependencies are injected, not hardcoded
- [ ] Context is properly propagated
- [ ] Resources are properly cleaned up (defer statements)

## Anti-Patterns to Avoid

1. **God Objects**: Classes/structs that do too much
2. **Spaghetti Code**: Tangled control flow
3. **Copy-Paste Programming**: Violates DRY
4. **Magic Numbers**: Use named constants
5. **Long Parameter Lists**: Use structs or functional options
6. **Ignoring Errors**: Always handle or propagate
7. **Premature Optimization**: Profile first
8. **Global State**: Use dependency injection
9. **Tight Coupling**: Depend on interfaces
10. **Missing Tests**: Every feature needs tests

## Implementation Examples

### Cache Implementation Pattern
```go
// Follow interface segregation - separate read and write operations
type CacheReader interface {
    Get(ctx context.Context, key string) (interface{}, error)
    Query(ctx context.Context, text string, threshold float64) (*QueryResult, error)
}

type CacheWriter interface {
    Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error
    Delete(ctx context.Context, key string) error
}

type Cache interface {
    CacheReader
    CacheWriter
    io.Closer
}
```

### Storage Backend Pattern
```go
// Follow dependency inversion - depend on abstractions
type Storage interface {
    Store(ctx context.Context, embedding []float64, text string, metadata map[string]interface{}) error
    Search(ctx context.Context, embedding []float64, threshold float64, limit int) ([]SearchResult, error)
    Delete(ctx context.Context, id string) error
    Close() error
}

// Each backend implements the interface
type PostgreSQLStore struct {
    db *sql.DB
    // ... other fields
}

func (p *PostgreSQLStore) Store(ctx context.Context, embedding []float64, text string, metadata map[string]interface{}) error {
    // Implementation following KISS principle
}
```

### Error Handling Pattern
```go
// Custom error types for better error handling
type ErrNotFound struct {
    Key string
}

func (e ErrNotFound) Error() string {
    return fmt.Sprintf("key not found: %s", e.Key)
}

// Wrap errors with context
func (c *Cache) Get(ctx context.Context, key string) (interface{}, error) {
    value, err := c.storage.Get(ctx, key)
    if err != nil {
        return nil, fmt.Errorf("cache get failed: %w", err)
    }
    return value, nil
}
```

### Concurrent Operations Pattern
```go
// Use goroutines with proper synchronization
func (o *Orchestrator) QueryMultipleAgents(ctx context.Context, query string) ([]Response, error) {
    var wg sync.WaitGroup
    responses := make([]Response, len(o.agents))
    errors := make([]error, len(o.agents))
    
    for i, agent := range o.agents {
        wg.Add(1)
        go func(idx int, a Agent) {
            defer wg.Done()
            resp, err := a.Query(ctx, query)
            responses[idx] = resp
            errors[idx] = err
        }(i, agent)
    }
    
    wg.Wait()
    // Process responses and errors
}
```

## Performance Optimization Guidelines

1. **Embedding Caching**: Cache frequently used embeddings to reduce API calls
2. **Batch Operations**: Process multiple items together when possible
3. **Connection Pooling**: Reuse database and Redis connections
4. **Lazy Loading**: Load data only when needed
5. **Efficient Serialization**: Use protocol buffers or msgpack for large data
6. **Index Optimization**: Create appropriate indexes for vector searches
7. **Query Optimization**: Use prepared statements and limit result sets

## Security Best Practices

1. **API Key Management**: Never hardcode API keys
2. **Input Validation**: Validate all user inputs
3. **SQL Injection Prevention**: Use parameterized queries
4. **Rate Limiting**: Implement rate limiting for API endpoints
5. **Authentication**: Use proper authentication middleware
6. **Encryption**: Encrypt sensitive data at rest and in transit
7. **Audit Logging**: Log security-relevant events

## Monitoring and Observability

1. **Structured Logging**: Use consistent log formats
2. **Metrics Collection**: Export Prometheus metrics
3. **Distributed Tracing**: Implement OpenTelemetry spans
4. **Health Checks**: Provide comprehensive health endpoints
5. **Performance Monitoring**: Track response times and throughput
6. **Error Tracking**: Aggregate and alert on errors
7. **Resource Monitoring**: Monitor CPU, memory, and connections

## Code Generation Attribution

This repository contains code and documentation generated with assistance from Claude Code (claude.ai/code). The AI assistance includes:
- Code refactoring and organization
- Documentation writing and updates
- Deployment script creation
- Configuration file generation
- README and guide creation

All generated content has been reviewed and integrated as part of the normal development process.