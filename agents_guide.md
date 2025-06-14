# AGENTS.md - Go Development Instructions for AI Agents

This file provides comprehensive instructions for AI agents (such as OpenAI Codex) working on this Go-based semantic cache system. All code changes must adhere strictly to Google's Go Style Guide and the patterns established in this codebase.

## Code Style and Formatting Requirements

### Mandatory Formatting
**All Go source files MUST conform to `gofmt` output.** This is non-negotiable.

```bash
# You MUST run these commands before committing any Go code:
gofmt -w .
go vet ./...
go test ./...
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
    "github.com/go-redis/redis/v8"
    "gorm.io/gorm"

    // 3. Local packages
    "github.com/your-org/semantic-cache/internal/cache"
    "github.com/your-org/semantic-cache/internal/storage"
)
```

### Package Structure (Must Follow)
```
cmd/                    # Application entry points
├── server/            # Main server application
└── cli/               # Command-line tools

internal/              # Private application code
├── cache/            # Core caching logic
├── storage/          # Storage implementations
│   ├── memory/       # In-memory cache
│   ├── redis/        # Redis implementation
│   └── postgres/     # PostgreSQL implementation
├── embedding/        # OpenAI embedding client
├── similarity/       # Similarity algorithms
└── config/           # Configuration management

pkg/                   # Public library code
├── client/           # Public cache client
└── types/            # Shared types

api/                   # API definitions
├── rest/             # REST handlers
└── grpc/             # gRPC services
```

### File Organization
- Keep files focused and reasonably sized (typically < 1000 lines)
- One primary type per file when possible
- Group related functionality together

## Error Handling (Mandatory Patterns)

### Error Return Values
- **Always handle errors explicitly**
- **Use `error` as the last return value**
- **Never ignore errors with `_`**

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

### Error Handling Patterns
- Handle errors early and return immediately
- Use `fmt.Errorf` with `%w` verb for error wrapping
- Avoid `else` blocks after error returns

```go
// CORRECT - Early return pattern
func (c *Cache) ValidateAndStore(ctx context.Context, key string, value []byte) error {
    if len(key) == 0 {
        return fmt.Errorf("key cannot be empty")
    }
    
    if len(value) == 0 {
        return fmt.Errorf("value cannot be empty")
    }
    
    if err := c.store(ctx, key, value); err != nil {
        return fmt.Errorf("failed to store: %w", err)
    }
    
    return nil
}

// INCORRECT - Nested if/else
func (c *Cache) ValidateAndStore(ctx context.Context, key string, value []byte) error {
    if len(key) == 0 {
        return fmt.Errorf("key cannot be empty")
    } else {
        if len(value) == 0 {
            return fmt.Errorf("value cannot be empty")
        } else {
            if err := c.store(ctx, key, value); err != nil {
                return fmt.Errorf("failed to store: %w", err)
            } else {
                return nil
            }
        }
    }
}
```

### Structured Errors
Create structured errors for better error handling:

```go
// CORRECT
type CacheError struct {
    Operation string
    Key       string
    Cause     error
}

func (e *CacheError) Error() string {
    return fmt.Sprintf("cache operation '%s' failed for key '%s': %v", 
        e.Operation, e.Key, e.Cause)
}

func (c *Cache) Get(ctx context.Context, key string) (interface{}, error) {
    value, err := c.storage.Get(ctx, key)
    if err != nil {
        return nil, &CacheError{
            Operation: "get",
            Key:       key,
            Cause:     err,
        }
    }
    return value, nil
}
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

### Channel Directions
**Specify channel directions to make data flow clear:**

```go
// CORRECT
func (p *Processor) StartPipeline(ctx context.Context) {
    queries := make(chan Query, 100)
    embeddings := make(chan Embedding, 100)
    
    go p.receiveQueries(ctx, queries)           // send-only: chan<- Query
    go p.processEmbeddings(ctx, queries, embeddings) // receive and send
    go p.storeResults(ctx, embeddings)          // receive-only: <-chan Embedding
}

func (p *Processor) receiveQueries(ctx context.Context, output chan<- Query) {
    defer close(output)
    // implementation
}
```

### Synchronous vs Asynchronous
Prefer synchronous functions for clarity:

```go
// CORRECT - Synchronous with clear error handling
func (nn *NeuralNetwork) Train(data TrainingSet) error {
    for epoch := 0; epoch < nn.config.MaxEpochs; epoch++ {
        if err := nn.trainEpoch(data); err != nil {
            return fmt.Errorf("epoch %d failed: %w", epoch, err)
        }
    }
    return nil
}

// Usage: caller controls concurrency if needed
go func() {
    if err := neuralNet.Train(trainingData); err != nil {
        log.Printf("Training failed: %v", err)
    }
}()

// INCORRECT - Hidden concurrency makes error handling difficult
func (nn *NeuralNetwork) TrainAsync(data TrainingSet, callback func(error)) {
    go func() {
        // How do we handle errors here?
        for epoch := 0; epoch < nn.config.MaxEpochs; epoch++ {
            nn.trainEpoch(data) // Errors lost
        }
        callback(nil)
    }()
}
```

## Testing Requirements (Must Implement)

### Table-Driven Tests
**You MUST use table-driven tests for comprehensive coverage:**

```go
func TestSimilarityCalculator_CosineSimilarity(t *testing.T) {
    tests := []struct {
        name     string
        vectorA  []float64
        vectorB  []float64
        expected float64
        wantErr  bool
    }{
        {
            name:     "identical_vectors",
            vectorA:  []float64{1.0, 0.0, 0.0},
            vectorB:  []float64{1.0, 0.0, 0.0},
            expected: 1.0,
            wantErr:  false,
        },
        {
            name:     "orthogonal_vectors",
            vectorA:  []float64{1.0, 0.0},
            vectorB:  []float64{0.0, 1.0},
            expected: 0.0,
            wantErr:  false,
        },
        {
            name:     "zero_vector",
            vectorA:  []float64{0.0, 0.0},
            vectorB:  []float64{1.0, 0.0},
            expected: 0.0,
            wantErr:  true,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            calc := NewSimilarityCalculator()
            got, err := calc.CosineSimilarity(tt.vectorA, tt.vectorB)
            
            if (err != nil) != tt.wantErr {
                t.Errorf("CosineSimilarity() error = %v, wantErr %v", err, tt.wantErr)
                return
            }
            
            if !tt.wantErr && math.Abs(got-tt.expected) > 1e-9 {
                t.Errorf("CosineSimilarity() = %v, want %v", got, tt.expected)
            }
        })
    }
}
```

### Test Helpers
**Create focused test helpers with `t.Helper()`:**

```go
func newTestCache(t *testing.T, options ...CacheOption) *Cache {
    t.Helper()
    
    config := DefaultConfig()
    for _, opt := range options {
        opt(&config)
    }
    
    cache, err := NewCache(config)
    if err != nil {
        t.Fatalf("failed to create test cache: %v", err)
    }
    
    return cache
}
```

### Mock Interfaces
Use interfaces for testability:

```go
// CORRECT - Define interfaces for external dependencies
type Storage interface {
    Get(ctx context.Context, key string) (interface{}, error)
    Set(ctx context.Context, key string, value interface{}) error
    Delete(ctx context.Context, key string) error
}

type MockStorage struct {
    data map[string]interface{}
    mu   sync.RWMutex
}

func (m *MockStorage) Get(ctx context.Context, key string) (interface{}, error) {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    value, exists := m.data[key]
    if !exists {
        return nil, fmt.Errorf("key not found: %s", key)
    }
    return value, nil
}

func (m *MockStorage) Set(ctx context.Context, key string, value interface{}) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    if m.data == nil {
        m.data = make(map[string]interface{})
    }
    m.data[key] = value
    return nil
}

func TestCache_BasicOperations(t *testing.T) {
    storage := &MockStorage{}
    cache := NewCacheWithStorage(storage)
    
    ctx := context.Background()
    
    // Test Set
    err := cache.Set(ctx, "test-key", "test-value")
    require.NoError(t, err)
    
    // Test Get
    value, err := cache.Get(ctx, "test-key")
    require.NoError(t, err)
    assert.Equal(t, "test-value", value)
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

### Function Documentation
**Document all exported functions with examples:**

```go
// CosineSimilarity calculates the cosine similarity between two vectors.
//
// Cosine similarity measures the cosine of the angle between two vectors,
// providing a value between -1 and 1 where 1 indicates identical direction,
// 0 indicates orthogonal vectors, and -1 indicates opposite directions.
//
// Returns an error if vectors have different dimensions or if either vector
// is the zero vector (which would make the calculation undefined).
//
// Example:
//
//   calc := similarity.NewCalculator()
//   sim, err := calc.CosineSimilarity(
//       []float64{1.0, 0.0, 0.0},
//       []float64{1.0, 0.0, 0.0},
//   )
//   if err != nil {
//       log.Fatal(err)
//   }
//   fmt.Printf("Similarity: %.2f\n", sim) // Output: Similarity: 1.00
func (c *Calculator) CosineSimilarity(a, b []float64) (float64, error) {
    if len(a) != len(b) {
        return 0, fmt.Errorf("vectors must have same dimension: got %d and %d", len(a), len(b))
    }
    
    if len(a) == 0 {
        return 0, fmt.Errorf("vectors cannot be empty")
    }
    
    var dotProduct, normA, normB float64
    for i := range a {
        dotProduct += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    
    if normA == 0 || normB == 0 {
        return 0, fmt.Errorf("cannot calculate similarity with zero vector")
    }
    
    return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB)), nil
}
```

### Type Documentation
Document important types and their usage:

```go
// Cache represents a semantic cache for LLM responses.
//
// A Cache provides intelligent caching of Large Language Model responses using
// embedding-based similarity matching. Instead of requiring exact string matches,
// the cache can identify conceptually similar queries and return cached responses,
// significantly reducing API costs and response latency.
//
// The Cache supports multiple storage backends and similarity algorithms,
// making it suitable for various deployment scenarios from development
// to high-scale production environments.
//
// Example:
//
//   cache := &Cache{
//       storage:    postgres.NewStorage(db),
//       similarity: cosine.NewCalculator(),
//       embedder:   openai.NewEmbedder(apiKey),
//       threshold:  0.85,
//   }
//   
//   result, found, err := cache.Get(ctx, "Explain neural networks")
//   if err != nil {
//       log.Fatal(err)
//   }
//   
//   if !found {
//       result = callLLM("Explain neural networks")
//       cache.Set(ctx, "Explain neural networks", result)
//   }
type Cache struct {
    // storage handles persistent storage of cached responses
    storage Storage
    
    // similarity calculates semantic similarity between queries
    similarity SimilarityCalculator
    
    // embedder converts text queries to vector embeddings
    embedder Embedder
    
    // threshold is the minimum similarity score for cache hits
    threshold float64
    
    // config holds cache configuration
    config Config
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

### String Building
**Use `strings.Builder` for concatenation:**

```go
// CORRECT
func (r *Report) GenerateReport(stats []Stat) string {
    var b strings.Builder
    b.WriteString("Cache Performance Report\n")
    b.WriteString("========================\n")
    
    for _, stat := range stats {
        fmt.Fprintf(&b, "Metric: %s, Value: %.2f\n", stat.Name, stat.Value)
    }
    
    return b.String()
}

// INCORRECT - String concatenation
func (r *Report) GenerateReport(stats []Stat) string {
    result := "Cache Performance Report\n"
    result += "========================\n"
    
    for _, stat := range stats {
        result += fmt.Sprintf("Metric: %s, Value: %.2f\n", stat.Name, stat.Value)
    }
    
    return result
}
```

### Avoid Copying Large Structs
Use pointers for large structs:

```go
// CORRECT - Use pointers for large structs
type CacheEntry struct {
    Query      string
    Response   string
    Embedding  []float64 // potentially large
    Metadata   map[string]interface{}
    Timestamp  time.Time
}

func (c *Cache) StoreEntry(entry *CacheEntry) error {
    return c.storage.Store(entry)
}

// INCORRECT - Copying large structs
func (c *Cache) StoreEntry(entry CacheEntry) error { // copies entire struct
    return c.storage.Store(&entry)
}
```

## Configuration Patterns (Must Use)

### Functional Options Pattern
**Use functional options for complex configuration:**

```go
// CORRECT - Functional options
type CacheConfig struct {
    MaxSize           int
    SimilarityThreshold float64
    TTL               time.Duration
    StorageBackend    string
}

type CacheOption func(*CacheConfig)

func WithMaxSize(size int) CacheOption {
    return func(c *CacheConfig) {
        c.MaxSize = size
    }
}

func WithSimilarityThreshold(threshold float64) CacheOption {
    return func(c *CacheConfig) {
        c.SimilarityThreshold = threshold
    }
}

func NewCache(options ...CacheOption) (*Cache, error) {
    config := CacheConfig{
        MaxSize:           1000,
        SimilarityThreshold: 0.85,
        TTL:               time.Hour,
        StorageBackend:    "memory",
    }
    
    for _, opt := range options {
        opt(&config)
    }
    
    return &Cache{config: config}, nil
}

// Usage
cache, err := NewCache(
    WithMaxSize(5000),
    WithSimilarityThreshold(0.9),
)
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

### Structured Logging
Use structured logging with context:

```go
import "log/slog"

func (c *Cache) ProcessQuery(ctx context.Context, query string) (*Response, error) {
    logger := slog.With(
        "component", "cache",
        "operation", "process_query",
        "query_hash", hashQuery(query),
    )
    
    logger.InfoContext(ctx, "Processing query", 
        "query_length", len(query),
    )
    
    result, err := c.processQueryInternal(ctx, query)
    if err != nil {
        logger.ErrorContext(ctx, "Query processing failed",
            "error", err,
        )
        return nil, err
    }
    
    logger.InfoContext(ctx, "Query processed successfully",
        "cache_hit", result.FromCache,
        "similarity_score", result.SimilarityScore,
    )
    
    return result, nil
}
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

# 7. Run staticcheck if available
if command -v staticcheck &> /dev/null; then
    staticcheck ./...
fi

# 8. Check for common issues
go mod tidy
go mod verify

# 9. Build the project to ensure it compiles
go build ./cmd/server
go build ./cmd/cli

# 10. Check for inefficient assignments
if command -v ineffassign &> /dev/null; then
    ineffassign ./...
fi

# 11. Check for misspellings
if command -v misspell &> /dev/null; then
    misspell -error .
fi
```

## Agent-Specific Instructions

### When Working with OpenAI API Integration
- Always handle rate limiting gracefully with exponential backoff
- Implement proper timeout handling for API requests
- Use context for request cancellation
- Never log API keys or sensitive data
- Implement circuit breaker pattern for resilience

```go
// CORRECT - Proper OpenAI client with resilience
type OpenAIClient struct {
    client      *http.Client
    apiKey      string
    baseURL     string
    rateLimiter *rate.Limiter
    circuitBreaker *CircuitBreaker
}

func (c *OpenAIClient) GetEmbedding(ctx context.Context, text string) ([]float64, error) {
    // Wait for rate limiter
    if err := c.rateLimiter.Wait(ctx); err != nil {
        return nil, fmt.Errorf("rate limiter: %w", err)
    }
    
    // Check circuit breaker
    if !c.circuitBreaker.Allow() {
        return nil, fmt.Errorf("circuit breaker open")
    }
    
    // Make API call with timeout
    ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
    defer cancel()
    
    embedding, err := c.makeAPICall(ctx, text)
    if err != nil {
        c.circuitBreaker.RecordFailure()
        return nil, err
    }
    
    c.circuitBreaker.RecordSuccess()
    return embedding, nil
}
```

### When Working with PostgreSQL/pgvector
- Use prepared statements for all queries
- Implement proper connection pooling
- Handle database migrations carefully
- Use appropriate vector indexes (HNSW or IVFFlat)

```go
// CORRECT - Proper PostgreSQL implementation
type PostgreSQLStorage struct {
    db *sql.DB
    stmts map[string]*sql.Stmt
}

func (s *PostgreSQLStorage) init() error {
    queries := map[string]string{
        "insert": `INSERT INTO cache_entries (query_hash, query_text, response, embedding, created_at) 
                   VALUES ($1, $2, $3, $4, $5)`,
        "search": `SELECT query_text, response, embedding <=> $1 AS distance 
                   FROM cache_entries 
                   WHERE embedding <=> $1 < $2 
                   ORDER BY distance 
                   LIMIT $3`,
    }
    
    s.stmts = make(map[string]*sql.Stmt)
    for name, query := range queries {
        stmt, err := s.db.Prepare(query)
        if err != nil {
            return fmt.Errorf("failed to prepare %s statement: %w", name, err)
        }
        s.stmts[name] = stmt
    }
    
    return nil
}

func (s *PostgreSQLStorage) FindSimilar(ctx context.Context, embedding []float64, threshold float64, limit int) ([]*CacheEntry, error) {
    rows, err := s.stmts["search"].QueryContext(ctx, pq.Array(embedding), threshold, limit)
    if err != nil {
        return nil, fmt.Errorf("failed to query similar entries: %w", err)
    }
    defer rows.Close()
    
    var entries []*CacheEntry
    for rows.Next() {
        entry := &CacheEntry{}
        var embeddingArray pq.Float64Array
        var distance float64
        
        err := rows.Scan(&entry.QueryText, &entry.Response, &distance)
        if err != nil {
            return nil, fmt.Errorf("failed to scan row: %w", err)
        }
        
        entry.SimilarityScore = 1.0 - distance // Convert distance to similarity
        entries = append(entries, entry)
    }
    
    return entries, rows.Err()
}
```

### When Working with Redis
- Handle Redis connection failures gracefully
- Implement circuit breaker pattern for Redis operations
- Use Redis pipelines for batch operations
- Set appropriate TTLs for cached data

```go
// CORRECT - Resilient Redis implementation
type RedisStorage struct {
    client   redis.UniversalClient
    pipeline redis.Pipeliner
    circuit  *CircuitBreaker
}

func (s *RedisStorage) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
    if !s.circuit.Allow() {
        return fmt.Errorf("redis circuit breaker open")
    }
    
    err := s.client.Set(ctx, key, value, ttl).Err()
    if err != nil {
        s.circuit.RecordFailure()
        return fmt.Errorf("redis set failed: %w", err)
    }
    
    s.circuit.RecordSuccess()
    return nil
}

func (s *RedisStorage) GetBatch(ctx context.Context, keys []string) (map[string]string, error) {
    if !s.circuit.Allow() {
        return nil, fmt.Errorf("redis circuit breaker open")
    }
    
    pipe := s.client.Pipeline()
    commands := make([]*redis.StringCmd, len(keys))
    
    for i, key := range keys {
        commands[i] = pipe.Get(ctx, key)
    }
    
    _, err := pipe.Exec(ctx)
    if err != nil && err != redis.Nil {
        s.circuit.RecordFailure()
        return nil, fmt.Errorf("redis pipeline exec failed: %w", err)
    }
    
    results := make(map[string]string)
    for i, cmd := range commands {
        if cmd.Err() == nil {
            results[keys[i]] = cmd.Val()
        }
    }
    
    s.circuit.RecordSuccess()
    return results, nil
}
```

### When Working with Tests
- Always add tests for new functionality
- Maintain test coverage above 80%
- Use table-driven tests for comprehensive coverage
- Mock external dependencies properly

```go
// CORRECT - Comprehensive test with mocks
func TestCache_Integration(t *testing.T) {
    tests := []struct {
        name           string
        setupMocks     func(*MockStorage, *MockEmbedder)
        query          string
        expectedResult *CacheResult
        expectedError  string
    }{
        {
            name: "cache_hit_exact_match",
            setupMocks: func(storage *MockStorage, embedder *MockEmbedder) {
                storage.On("Get", mock.Anything, "exact_match_key").Return(&CacheEntry{
                    QueryText: "What is machine learning?",
                    Response:  "Machine learning is...",
                }, nil)
            },
            query: "What is machine learning?",
            expectedResult: &CacheResult{
                Response:   "Machine learning is...",
                FromCache:  true,
                Similarity: 1.0,
            },
        },
        {
            name: "cache_miss_similarity_below_threshold",
            setupMocks: func(storage *MockStorage, embedder *MockEmbedder) {
                embedder.On("GetEmbedding", mock.Anything, "What is deep learning?").Return([]float64{0.1, 0.2, 0.3}, nil)
                storage.On("FindSimilar", mock.Anything, []float64{0.1, 0.2, 0.3}, 0.85, 5).Return([]*CacheEntry{
                    {QueryText: "What is ML?", Response: "ML is...", SimilarityScore: 0.7},
                }, nil)
            },
            query: "What is deep learning?",
            expectedResult: &CacheResult{
                Response:   "",
                FromCache:  false,
                Similarity: 0.0,
            },
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            storage := &MockStorage{}
            embedder := &MockEmbedder{}
            
            tt.setupMocks(storage, embedder)
            
            cache := NewCache(
                WithStorage(storage),
                WithEmbedder(embedder),
                WithSimilarityThreshold(0.85),
            )
            
            result, err := cache.Get(context.Background(), tt.query)
            
            if tt.expectedError != "" {
                assert.Error(t, err)
                assert.Contains(t, err.Error(), tt.expectedError)
            } else {
                assert.NoError(t, err)
                assert.Equal(t, tt.expectedResult, result)
            }
            
            storage.AssertExpectations(t)
            embedder.AssertExpectations(t)
        })
    }
}
```

### Common Patterns to Follow
1. **Repository Pattern**: Separate data access logic
2. **Dependency Injection**: Use interfaces and inject dependencies
3. **Configuration**: Use functional options pattern
4. **Logging**: Use structured logging with context
5. **Metrics**: Export Prometheus metrics for monitoring
6. **Tracing**: Use OpenTelemetry for distributed tracing
7. **Circuit Breaker**: Implement resilience patterns for external services
8. **Rate Limiting**: Control resource usage and API calls

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

## Performance Impact
- [ ] No performance regression expected
- [ ] Performance benchmarks run (if applicable)
- [ ] Memory usage checked

## Breaking Changes
- [ ] No breaking changes
- [ ] Breaking changes documented in commit message

## Checklist
- [ ] Code formatted with `gofmt`
- [ ] `go vet` passes
- [ ] All tests pass
- [ ] Test coverage > 80%
- [ ] Documentation updated
- [ ] AGENTS.md guidelines followed
```

### Commit Message Format
```
package: brief description in imperative mood

Longer explanation if needed. Explain what and why, not how.
Include any breaking changes and migration instructions.

Fixes #123
```

## Summary

This comprehensive guide ensures all Go code in this semantic cache system follows Google's Go Style Guide while being optimized for AI agents and semantic caching workloads. Key requirements:

1. **Formatting**: Always use `gofmt` and `go vet`
2. **Naming**: Strict camelCase, no underscores, consistent receivers
3. **Error Handling**: Explicit error handling, early returns, structured errors
4. **Concurrency**: Context propagation, explicit goroutine management
5. **Testing**: Table-driven tests, mocks, >80% coverage
6. **Documentation**: Comprehensive package and function docs
7. **Performance**: Pre-allocation, avoid copying, efficient string building
8. **Observability**: OpenTelemetry tracing, structured logging
9. **Resilience**: Circuit breakers, rate limiting, graceful degradation

**Remember**: This is a production system handling potentially sensitive data. Always prioritize security, performance, and maintainability. When in doubt, follow the principle of least surprise and make the code as readable as possible.