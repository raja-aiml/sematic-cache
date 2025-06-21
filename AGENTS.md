# AGENTS.md - Go Development Instructions for AI Agents

This file provides comprehensive instructions for AI agents (such as OpenAI Codex) working on this Go-based semantic cache system. All code changes must adhere strictly to Google's Go Style Guide and the patterns established in this codebase.

## 🚨 CRITICAL: MUST READ CLAUDE.md FIRST 🚨

**MANDATORY**: Before making ANY code changes, you MUST load and follow CLAUDE.md which contains:
- Approved technology stack
- SDK-first development rules
- Interface compliance requirements
- Strict architectural guidelines

## Code Style and Formatting Requirements

### Mandatory Formatting
**All Go source files MUST conform to `gofmt` output.** This is non-negotiable.

```bash
# You MUST run these commands before committing any Go code:
gofmt -w .
go vet ./...
go test ./...
go mod tidy  # MANDATORY after dependency changes
```

### Technology Stack Compliance

**Web Framework:**
```go
// CORRECT - Using approved Gin framework
import "github.com/gin-gonic/gin"

router := gin.Default()
router.GET("/health", handleHealth)

// INCORRECT - Using non-approved frameworks
import "github.com/gofiber/fiber/v2"  // ❌ NOT APPROVED
import "github.com/labstack/echo/v4"  // ❌ NOT APPROVED
```

**Docker Operations:**
```go
// CORRECT - Using Docker SDK
import (
    "github.com/docker/docker/client"
    "github.com/docker/docker/api/types"
)

client, err := client.NewClientWithOpts(client.FromEnv)
if err != nil {
    return fmt.Errorf("failed to create docker client: %w", err)
}
defer client.Close()

// INCORRECT - Using CLI commands
cmd := exec.Command("docker", "build", "-t", tag, ".")  // ❌ FORBIDDEN
```

### Mixed Caps Naming (Strictly Enforced)
Use `MixedCaps` or `mixedCaps` (camelCase) - **never underscores**:

```go
// CORRECT
const MaxRetries = 3
const maxBufferSize = 1024
type NeuralNetwork struct{}
func (nn *NeuralNetwork) ForwardPass() {}

// INCORRECT - Never use underscores
const MAX_RETRIES = 3
const max_buffer_size = 1024
type Neural_Network struct{}
func (nn *Neural_Network) forward_pass() {}
```

### Line Length
**No fixed line length limit.** If a line feels too long, prefer refactoring over breaking it. Focus on clarity over arbitrary length constraints.

```go
// CORRECT - Keep related logic together
agent := NewAgent(WithBrain(brain), WithMemory(memory), WithSensors(sensors))

// INCORRECT - Arbitrary line breaks that hurt readability
agent := NewAgent(WithBrain(brain),
    WithMemory(memory), WithSensors(sensors))
```

## Naming Conventions (Mandatory Compliance)

### Package Names
Package names must be:
- **All lowercase**
- **Single words when possible**
- **Short and concise**
- **Related to what they provide**

```go
// CORRECT
package cache
package storage  
package embedding
package similarity

// INCORRECT
package cachePackage
package storage_layer
package embeddingUtils
```

### Variables and Functions

#### Variable Names
Variable name length should be proportional to scope:

```go
// CORRECT - Short names in small scopes
for i, item := range cache.items {
    if err := item.Process(); err != nil {
        return err
    }
}

// CORRECT - Descriptive names in larger scopes  
func ProcessSemanticQuery(embeddingVector []float64, similarityThreshold float64) (*QueryResult, error)

// INCORRECT - Underscores not allowed
func process_query(embedding_vector []float64) error
func ProcessQuery(embedding_vector []float64) error
```

#### Function Names
- Functions that **return something**: use noun-like names
- Functions that **do something**: use verb-like names

```go
// CORRECT - Returns something (noun-like)
func (c *Cache) CurrentSize() int
func (s *Storage) EmbeddingVector() []float64

// CORRECT - Does something (verb-like)  
func (c *Cache) Store(key string, value interface{})
func (s *Storage) Initialize() error
```

#### Receivers (Strict Requirements)
- **Must be 1-2 characters**
- **Must be consistent throughout the type**
- **Must be abbreviations of the type name**

```go
// CORRECT
func (c *Cache) Get(key string) (interface{}, bool) { }
func (c *Cache) Set(key string, value interface{}) { }

func (sm *SimilarityMatcher) Calculate(a, b []float64) float64 { }
func (sm *SimilarityMatcher) SetThreshold(threshold float64) { }

// INCORRECT - Inconsistent receiver names
func (cache *Cache) Get(key string) (interface{}, bool) { }
func (c *Cache) Set(key string, value interface{}) { }
```

#### Constants
Use `MixedCaps` for constants (not SCREAMING_SNAKE_CASE):

```go
// CORRECT
const MaxRetries = 3
const DefaultSimilarityThreshold = 0.85
const (
    StatusIdle = iota
    StatusProcessing
    StatusComplete
)

// INCORRECT
const MAX_RETRIES = 3
const default_similarity_threshold = 0.85
```

## Code Organization Requirements

### Import Organization (Exactly Three Blocks)
**You MUST group imports into exactly three blocks in this order:**

```go
import (
    // 1. Standard library
    "context"
    "fmt"
    "time"

    // 2. Third-party packages  
    "github.com/gin-gonic/gin"
    "github.com/go-redis/redis/v8"
    "gorm.io/gorm"

    // 3. Local packages
    "github.com/raja-aiml/sematic-cache/core"
    "github.com/raja-aiml/sematic-cache/storage"
)
```

### Package Structure (Must Follow)
```
cmd/                    # Application entry points
├── server/            # Main server application
└── cli/               # Command-line tools

core/                  # Core functionality
├── cache.go          # Core caching logic
├── agent.go          # Agent implementations
└── orchestrator.go   # Orchestration logic

storage/              # Storage implementations
├── factory.go        # Storage factory
├── redisstore.go     # Redis implementation
├── gormstore.go      # PostgreSQL implementation
└── composite.go      # Composite storage

server/               # HTTP server (Gin framework ONLY)
├── server.go         # Server implementation
├── server_test.go    # Server tests
└── README.md         # API documentation

openai/               # OpenAI SDK integration
├── client.go         # OpenAI client wrapper
└── client_test.go    # Client tests

config/               # Configuration management
└── config.go         # Viper-based configuration
```

### File Organization
- Keep files focused and reasonably sized (typically < 1000 lines)
- One primary type per file when possible
- Group related functionality together

## Interface Compliance (MANDATORY)

### Always Check Existing Interfaces First

```go
// STEP 1: Check existing interface
// Look in core/cache.go for:
type CacheBackend interface {
    Get(prompt string) (string, bool)
    GetModelInfo(prompt string) (modelName, modelID string, found bool)
    SetWithModel(prompt string, embedding []float32, answer, modelName, modelID string)
    SetPromptWithModel(prompt, answer, modelName, modelID string) error
    GetTopKByEmbedding(embed []float32, k int) []QueryResult
    Flush()
    Stats() (hits, misses uint64, hitRate float64)
}

// STEP 2: Implement correctly
type MyCache struct {
    // fields
}

// Implement ALL methods
func (c *MyCache) Get(prompt string) (string, bool) { /* ... */ }
func (c *MyCache) GetModelInfo(prompt string) (string, string, bool) { /* ... */ }
// ... implement all other methods

// STEP 3: Verify at compile time
var _ CacheBackend = (*MyCache)(nil)
```

## Error Handling (Mandatory Patterns)

### Error Return Values
- **Always handle errors explicitly**
- **Use `error` as the last return value**
- **Never ignore errors with `_`**
- **Always wrap errors with context**

```go
// CORRECT
func (c *Cache) ProcessQuery(ctx context.Context, query string) (*Result, error) {
    embedding, err := c.embedder.GetEmbedding(ctx, query)
    if err != nil {
        return nil, fmt.Errorf("failed to get embedding: %w", err)
    }
    
    result, err := c.storage.FindSimilar(ctx, embedding)
    if err != nil {
        return nil, fmt.Errorf("failed to find similar: %w", err)
    }
    
    return result, nil
}

// INCORRECT - Ignoring errors
func (c *Cache) ProcessQuery(ctx context.Context, query string) *Result {
    embedding, _ := c.embedder.GetEmbedding(ctx, query)  // BAD: ignoring error
    result, _ := c.storage.FindSimilar(ctx, embedding)   // BAD: ignoring error
    return result
}
```

## SDK Usage Requirements (STRICTLY ENFORCED)

### Docker Operations
```go
// CORRECT - Using Docker SDK
import (
    "github.com/docker/docker/client"
    "github.com/docker/docker/api/types"
)

type DockerBuilder struct {
    client *client.Client
}

func NewDockerBuilder() (*DockerBuilder, error) {
    cli, err := client.NewClientWithOpts(
        client.FromEnv,
        client.WithAPIVersionNegotiation(),
    )
    if err != nil {
        return nil, fmt.Errorf("failed to create docker client: %w", err)
    }
    return &DockerBuilder{client: cli}, nil
}

func (b *DockerBuilder) Build(ctx context.Context, path string, tag string) error {
    buildOpts := types.ImageBuildOptions{
        Tags: []string{tag},
        Dockerfile: "Dockerfile",
    }
    
    resp, err := b.client.ImageBuild(ctx, buildContext, buildOpts)
    if err != nil {
        return fmt.Errorf("failed to build image: %w", err)
    }
    defer resp.Body.Close()
    
    return nil
}

// INCORRECT - Using CLI commands
func BuildImage(tag string) error {
    cmd := exec.Command("docker", "build", "-t", tag, ".")  // ❌ FORBIDDEN
    return cmd.Run()
}
```

### Kubernetes Operations
```go
// CORRECT - Using client-go
import (
    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/tools/clientcmd"
)

func GetKubeClient() (*kubernetes.Clientset, error) {
    config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
    if err != nil {
        return nil, fmt.Errorf("failed to build config: %w", err)
    }
    
    clientset, err := kubernetes.NewForConfig(config)
    if err != nil {
        return nil, fmt.Errorf("failed to create client: %w", err)
    }
    
    return clientset, nil
}

// INCORRECT - Using kubectl commands
func ApplyManifest(file string) error {
    cmd := exec.Command("kubectl", "apply", "-f", file)  // ❌ FORBIDDEN
    return cmd.Run()
}
```

## Testing Requirements (Must Implement)

### Table-Driven Tests
**You MUST use table-driven tests for comprehensive coverage:**

```go
func TestCache_Get(t *testing.T) {
    tests := []struct {
        name      string
        setup     func(*mockStorage)
        prompt    string
        want      string
        wantFound bool
        wantErr   bool
    }{
        {
            name: "exact match found",
            setup: func(m *mockStorage) {
                m.data["hello"] = "world"
            },
            prompt:    "hello",
            want:      "world",
            wantFound: true,
            wantErr:   false,
        },
        {
            name:      "not found",
            setup:     func(m *mockStorage) {},
            prompt:    "missing",
            want:      "",
            wantFound: false,
            wantErr:   false,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            storage := &mockStorage{data: make(map[string]string)}
            tt.setup(storage)
            
            cache := &Cache{storage: storage}
            got, found := cache.Get(tt.prompt)
            
            assert.Equal(t, tt.want, got)
            assert.Equal(t, tt.wantFound, found)
        })
    }
}
```

### Using testify for Assertions
```go
import (
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "github.com/stretchr/testify/mock"
)

// Use require for critical assertions that should stop the test
require.NoError(t, err)
require.NotNil(t, result)

// Use assert for non-critical assertions
assert.Equal(t, expected, actual)
assert.Contains(t, output, "expected text")
```

## Concurrency Patterns (Strictly Required)

### Context Propagation
**Always pass `context.Context` as the first parameter:**

```go
// CORRECT
func (c *Cache) Get(ctx context.Context, key string) (interface{}, error)
func (s *Storage) Store(ctx context.Context, embedding []float64, response string) error

// INCORRECT
func (c *Cache) Get(key string) (interface{}, error)
func (s *Storage) Store(embedding []float64, response string) error
```

### Goroutine Management
**Make goroutine lifetimes explicit:**

```go
// CORRECT - Explicit lifetime management
func (s *Server) Start(ctx context.Context) error {
    var wg sync.WaitGroup
    
    wg.Add(1)
    go func() {
        defer wg.Done()
        s.cacheWorker(ctx)
    }()
    
    wg.Add(1) 
    go func() {
        defer wg.Done()
        s.embeddingWorker(ctx)
    }()
    
    wg.Wait()
    return nil
}

// INCORRECT - No lifetime management
func (s *Server) Start() {
    go s.cacheWorker()    // When does this stop?
    go s.embeddingWorker() // No way to stop this
}
```

## Documentation Requirements (Must Follow)

### Package Documentation
**Every package MUST have comprehensive documentation:**

```go
// Package cache provides intelligent semantic caching for LLM responses.
//
// This package implements a multi-tier caching system that uses embedding-based
// similarity to identify conceptually similar queries. The main components are:
//
//   - Cache: The primary interface for storing and retrieving cached responses
//   - Storage: Pluggable storage backends (memory, Redis, PostgreSQL)
//   - Similarity: Vector similarity calculation algorithms
//
// Basic usage:
//
//   config := cache.DefaultConfig()
//   c, err := cache.New(config)
//   if err != nil {
//       log.Fatal(err)
//   }
//   
//   response, found, err := c.Get(ctx, "What is machine learning?")
//   if err != nil {
//       log.Fatal(err)
//   }
//   
//   if !found {
//       response = callLLM("What is machine learning?")
//       c.Set(ctx, "What is machine learning?", response)
//   }
package cache
```

## Configuration Patterns (Must Use)

### Using Viper for Configuration
```go
// CORRECT - Using approved Viper
import "github.com/spf13/viper"

func LoadConfig() (*Config, error) {
    viper.SetConfigName("config")
    viper.SetConfigType("yaml")
    viper.AddConfigPath(".")
    viper.AddConfigPath("./config")
    
    viper.SetEnvPrefix("CACHE")
    viper.AutomaticEnv()
    
    if err := viper.ReadInConfig(); err != nil {
        return nil, fmt.Errorf("failed to read config: %w", err)
    }
    
    var cfg Config
    if err := viper.Unmarshal(&cfg); err != nil {
        return nil, fmt.Errorf("failed to unmarshal config: %w", err)
    }
    
    return &cfg, nil
}
```

### Using Cobra for CLI
```go
// CORRECT - Using approved Cobra
import "github.com/spf13/cobra"

var rootCmd = &cobra.Command{
    Use:   "cache-server",
    Short: "Semantic cache server for LLM responses",
    RunE: func(cmd *cobra.Command, args []string) error {
        return runServer(cmd.Context())
    },
}

func init() {
    rootCmd.PersistentFlags().String("config", "", "config file path")
    rootCmd.PersistentFlags().String("addr", ":8080", "server address")
}
```

## Performance Requirements

### Memory Management
**Pre-allocate slices and maps when size is known:**

```go
// CORRECT - Pre-allocate with known capacity
func (c *Cache) ProcessBatch(queries []string) ([]Result, error) {
    results := make([]Result, 0, len(queries)) // Pre-allocate capacity
    embeddings := make([][]float64, len(queries))
    
    for i, query := range queries {
        embedding, err := c.getEmbedding(query)
        if err != nil {
            return nil, err
        }
        embeddings[i] = embedding
    }
    
    return results, nil
}

// INCORRECT - No pre-allocation
func (c *Cache) ProcessBatch(queries []string) ([]Result, error) {
    var results []Result  // Will reallocate multiple times
    var embeddings [][]float64
    
    for _, query := range queries {
        // ... implementation
    }
    
    return results, nil
}
```

## OpenTelemetry Integration (Required)

### Context Propagation
**Always propagate OpenTelemetry context:**

```go
import (
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/trace"
)

func (c *Cache) Get(ctx context.Context, query string) (*Response, error) {
    tracer := otel.Tracer("semantic-cache")
    ctx, span := tracer.Start(ctx, "cache.get",
        trace.WithAttributes(
            attribute.String("query.hash", hashQuery(query)),
            attribute.Int("query.length", len(query)),
        ),
    )
    defer span.End()
    
    // Check memory cache first
    if result, found := c.checkMemoryCache(ctx, query); found {
        span.SetAttributes(attribute.String("cache.layer", "memory"))
        return result, nil
    }
    
    // Check Redis cache
    result, err := c.checkRedisCache(ctx, query)
    if err != nil {
        span.RecordError(err)
        return nil, err
    }
    
    if result != nil {
        span.SetAttributes(attribute.String("cache.layer", "redis"))
        return result, nil
    }
    
    // Cache miss - will need to generate
    span.SetAttributes(attribute.Bool("cache.miss", true))
    return nil, ErrCacheMiss
}
```

## Web Server Implementation (Gin Framework ONLY)

### Creating HTTP Handlers with Gin
```go
// CORRECT - Using Gin framework
import "github.com/gin-gonic/gin"

type Server struct {
    cache  core.CacheBackend
    router *gin.Engine
}

func NewServer(cache core.CacheBackend) *Server {
    router := gin.Default()
    s := &Server{
        cache:  cache,
        router: router,
    }
    s.setupRoutes()
    return s
}

func (s *Server) setupRoutes() {
    api := s.router.Group("/api/v1")
    {
        api.GET("/health", s.handleHealth)
        api.POST("/cache/get", s.handleCacheGet)
        api.POST("/cache/set", s.handleCacheSet)
    }
}

func (s *Server) handleCacheGet(c *gin.Context) {
    var req GetRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    result, found := s.cache.Get(req.Prompt)
    if !found {
        c.JSON(http.StatusNotFound, gin.H{"error": "not found"})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{
        "prompt": req.Prompt,
        "result": result,
    })
}

// INCORRECT - Using non-approved framework
import "github.com/gofiber/fiber/v2"  // ❌ NOT APPROVED
```

## Programmatic Checks (MUST RUN)

**Before committing any code changes, you MUST run these commands and ensure they all pass:**

```bash
# 1. Format all Go code (MANDATORY)
gofmt -w .

# 2. Check for formatting issues  
if [ -n "$(gofmt -d .)" ]; then
    echo "Code is not properly formatted"
    exit 1
fi

# 3. Run go vet (MANDATORY)
go vet ./...

# 4. Run all tests (MANDATORY)
go test ./...

# 5. Run tests with race detection
go test -race ./...

# 6. Check test coverage (MUST be > 80%)
go test -cover ./... | grep -E "coverage: [0-9]+\.[0-9]+%" | awk '{print $2}' | grep -E "^[8-9][0-9]\.[0-9]+%|^100\.0%"

# 7. Clean up go.mod (MANDATORY after dependency changes)
go mod tidy
go mod verify

# 8. Build the project to ensure it compiles
go build ./...

# 9. Check that no CLI commands are used where SDK exists
! grep -r "exec.Command" --include="*.go" . | grep -E "(docker|kubectl|helm)"
```

## Common Pitfalls to Avoid

1. **Using CLI instead of SDK**: Always check for Go SDK first
2. **Wrong web framework**: Use Gin, not Fiber/Echo/Chi
3. **Ignoring interfaces**: Check existing interfaces before creating new types
4. **Poor error messages**: Always add context to errors
5. **Missing tests**: Every new feature needs tests
6. **Forgetting go mod tidy**: Always run after dependency changes
7. **Creating duplicate functionality**: Search codebase first
8. **Not checking CLAUDE.md**: Always load and follow architectural rules

## Agent-Specific Instructions

### When Working with OpenAI API Integration
- Use the official OpenAI Go SDK (github.com/openai/openai-go)
- Never use custom HTTP implementations
- Always handle rate limiting gracefully with exponential backoff
- Implement proper timeout handling for API requests
- Use context for request cancellation
- Never log API keys or sensitive data

```go
// CORRECT - Using official OpenAI SDK
import "github.com/openai/openai-go"

client := openai.NewClient(apiKey)
embedding, err := client.Embeddings.Create(ctx, openai.EmbeddingCreateParams{
    Model: openai.F(openai.EmbeddingModelTextEmbeddingAda002),
    Input: openai.F[openai.EmbeddingCreateParamsInputUnion](input),
})
```

### When Working with Docker
- ALWAYS use Docker SDK (github.com/docker/docker)
- NEVER use docker CLI commands
- Implement proper cleanup with defer
- Handle Docker daemon connection failures gracefully

### When Working with Kubernetes
- ALWAYS use client-go (k8s.io/client-go)
- NEVER use kubectl commands
- Use informers for watching resources
- Implement proper retry logic with backoff

## Pull Request Guidelines

### PR Title Format
- Use imperative mood: "Add feature" not "Added feature"
- Be specific: "Fix similarity calculation for zero vectors" not "Fix bug"
- Include affected package: "cache: improve memory allocation in batch processing"

### PR Description Template
```markdown
## Summary
Brief description of what this PR does.

## Changes Made
- Specific change 1 with file paths
- Specific change 2 with file paths

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing performed
- [ ] Test coverage maintained above 80%

## Checklist
- [ ] Code formatted with `gofmt`
- [ ] `go vet` passes
- [ ] All tests pass
- [ ] Test coverage > 80%
- [ ] go mod tidy run
- [ ] No CLI commands used where SDK exists
- [ ] Used approved tech stack only
- [ ] Documentation updated
- [ ] CLAUDE.md and AGENTS.md guidelines followed
```

## Summary

This comprehensive guide ensures all Go code in this semantic cache system follows Google's Go Style Guide while being optimized for AI agents and semantic caching workloads. Key requirements:

1. **Always load CLAUDE.md first** for architectural rules
2. **Use approved tech stack only** (Gin, Viper, Cobra, etc.)
3. **SDK over CLI always** (Docker SDK, Kubernetes client-go)
4. **Formatting**: Always use `gofmt` and `go vet`
5. **Naming**: Strict camelCase, no underscores, consistent receivers
6. **Error Handling**: Explicit error handling, early returns, structured errors
7. **Concurrency**: Context propagation, explicit goroutine management
8. **Testing**: Table-driven tests with testify, >80% coverage
9. **Documentation**: Comprehensive package and function docs
10. **Performance**: Pre-allocation, avoid copying, efficient string building

**Remember**: This is a production system handling potentially sensitive data. Always prioritize security, performance, and maintainability. When in doubt, follow the principle of least surprise and make the code as readable as possible.