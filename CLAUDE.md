# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with the semantic cache repository. It emphasizes architectural principles, code quality standards, and development best practices.

## 🚨 STRICT ARCHITECTURE RULES - MUST FOLLOW 🚨

### 1. ALWAYS USE SDK OVER CLI COMMANDS
- **MANDATORY**: Use official Go SDKs/libraries instead of CLI commands
- **FORBIDDEN**: Running external commands via exec/shell unless absolutely necessary
- **EXCEPTION**: Only when no SDK exists (e.g., k3d) and must be documented
- **Example**: Use Docker SDK instead of `docker` CLI, Kubernetes client-go instead of `kubectl`

### 2. APPROVED TECHNOLOGY STACK
**Web Frameworks:**
- ✅ **Gin** (github.com/gin-gonic/gin) - Primary web framework
- ❌ Fiber, Echo, Chi - Not approved

**Database/Storage:**
- ✅ **PostgreSQL** with pgx driver
- ✅ **Redis** with go-redis/redis/v8
- ✅ **GORM** for ORM when needed
- ❌ MongoDB, MySQL - Not approved without justification

**Testing:**
- ✅ **testify** (github.com/stretchr/testify) - Assertions and mocks
- ✅ **Table-driven tests** - Required pattern
- ✅ **httptest** - For HTTP testing
- ❌ Ginkgo, Gomega - Not approved

**Configuration:**
- ✅ **Viper** (github.com/spf13/viper) - Configuration management
- ✅ **Cobra** (github.com/spf13/cobra) - CLI framework
- ❌ Other config libraries without approval

**Observability:**
- ✅ **OpenTelemetry** - Tracing and metrics
- ✅ **Prometheus** client - Metrics
- ❌ Custom telemetry solutions

**Container/Orchestration:**
- ✅ **Docker SDK** (github.com/docker/docker)
- ✅ **Kubernetes client-go** (k8s.io/client-go)
- ✅ **Kustomize** libraries for manifest generation
- ❌ Docker CLI commands, kubectl exec

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
- **MANDATORY**: Verify compatibility with go.work workspace
- **FORBIDDEN**: Adding dependencies that duplicate existing functionality

## Quick Start Summary

### Essential Commands
```bash
gofmt -w .              # Format all code (MANDATORY before commit)
go test ./...           # Run all tests (MANDATORY before commit)
go vet ./...            # Static analysis (MANDATORY before commit)
go mod tidy             # Clean dependencies (MANDATORY after changes)
go run cmd/server/main.go -config config.yml  # Run server
```

### Git Commit Guidelines
When you need to quickly stage all changes and commit with an auto-generated message, use this instruction:
```
"git add, generate git commit message and commit"
```
This will:
1. Check git status to understand changes
2. Stage all modified and new files (limited to current directory when in iaac/iaac)
3. Generate a descriptive commit message following conventional commits format
4. Create the commit without any AI attribution

**IMPORTANT: Never include these lines in commit messages:**
```
🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

Note: When working in iaac/iaac, only changes within that directory will be staged and committed.

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
- `core/`: Core functionality (cache, agents, orchestrator)
- `storage/`: Storage backend implementations
- `server/`: HTTP API server (Gin framework)
- `openai/`: OpenAI integration
- `config/`: Configuration management
- `cmd/`: Application entry points
- `iaac/`: Infrastructure as Code
  - `blueprint/`: Kubernetes manifests
  - `infra/`: Go-based infrastructure tooling

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
    
    "github.com/raja-aiml/sematic-cache/core"
    "github.com/raja-aiml/sematic-cache/storage"
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

1. **Minimum 80% test coverage** for all packages
2. **Table-driven tests** for multiple test cases
3. **Mock external dependencies** using interfaces
4. **Test file naming**: `*_test.go` in the same package
5. **Benchmark tests** for performance-critical code

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
- [ ] Test coverage >= 80%
- [ ] No CLI commands used where SDK exists
- [ ] Interfaces properly implemented
- [ ] Errors wrapped with context
- [ ] Dependencies cleaned with `go mod tidy`
- [ ] Documentation updated if needed
- [ ] No sensitive data in logs or commits

## Common Pitfalls to Avoid

1. **Using CLI instead of SDK**: Always check for Go SDK first
2. **Ignoring interfaces**: Check existing interfaces before creating new types
3. **Poor error messages**: Always add context to errors
4. **Missing tests**: Every new feature needs tests
5. **Forgetting go mod tidy**: Always run after dependency changes
6. **Using wrong web framework**: Use Gin, not others
7. **Creating duplicate functionality**: Search codebase first

## Examples of Good Practices

### Using SDK instead of CLI
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

### Proper Interface Implementation
```go
// First, check the interface
type CacheBackend interface {
    Get(prompt string) (string, bool)
    Set(prompt string, value string) error
}

// Then implement correctly
type MyCache struct{}

func (c *MyCache) Get(prompt string) (string, bool) {
    // Implementation
}

func (c *MyCache) Set(prompt string, value string) error {
    // Implementation
}

// Verify at compile time
var _ CacheBackend = (*MyCache)(nil)
```

This document is the source of truth for all development decisions in this repository.