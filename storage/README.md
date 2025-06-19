# Storage Package

This package provides multiple storage backend implementations for the semantic cache system.

## Available Backends

### 1. In-Memory Cache (Default)
- **Type**: `"memory"` or empty string
- **Features**: Full support for all cache operations including vector search
- **Limitations**: Data is lost on restart
- **Use Case**: Development, testing, or when persistence is not required

### 2. Redis Cluster
- **Type**: `"redis"`
- **Features**: Distributed caching, TTL support, persistence
- **Limitations**: 
  - Does NOT support vector similarity search (`GetTopKByEmbedding` returns nil)
  - Requires Redis cluster configuration
- **Use Case**: High-performance distributed caching when vector search is not needed

### 3. PostgreSQL with pgvector (GORM)
- **Type**: `"gorm"`
- **Features**: Full vector similarity search, persistence, SQL queries
- **Limitations**: 
  - Requires PostgreSQL with pgvector extension
  - Slower than in-memory for simple key-value operations
- **Use Case**: When you need persistent storage with vector similarity search

### 4. Composite Multi-Tier Backend
- **Type**: `"composite"`
- **Features**: Combines multiple backends in a tiered architecture
  - Automatic fallback between tiers
  - Smart cache promotion
  - Per-tier metrics
  - Optimizes for speed, cost, and reliability
- **Use Case**: Production systems requiring optimal performance with multiple storage layers
- **Example**: Memory (L1) → Redis (L2) → PostgreSQL (L3)

## Configuration

### Memory Backend
```yaml
cache:
  type: "memory"  # or leave empty
  capacity: 1000
  eviction_policy: "LRU"
  min_similarity: 0.8
```

### Redis Backend
```yaml
cache:
  type: "redis"
  redis:
    addrs:
      - "localhost:6379"
    password: ""  # optional
```

### PostgreSQL Backend
```yaml
cache:
  type: "gorm"
# Also requires DATABASE_URL environment variable
```

### Composite Backend
```yaml
cache:
  type: "composite"
  min_similarity: 0.85
  composite:
    promote_on_hit: true
    tiers:
      - name: "memory-l1"
        type: "memory"
        priority: 1
        capacity: 1000
      - name: "redis-l2"
        type: "redis"
        priority: 2
        redis:
          addrs: ["localhost:6379"]
      - name: "postgres-l3"
        type: "gorm"
        priority: 3
```

See [COMPOSITE_BACKEND.md](../docs/COMPOSITE_BACKEND.md) for detailed documentation.

## Adding New Backends

To add a new storage backend:

1. Create your implementation file (e.g., `mongostore.go`)
2. Implement the `core.CacheBackend` interface
3. Register your backend in an `init()` function:

```go
func init() {
    Register("mongodb", newMongoBackend)
}

func newMongoBackend(cfg *config.Config, embedFunc core.EmbeddingFunc) (core.CacheBackend, error) {
    // Your implementation
}
```

That's it! No changes needed in main.go or other files.

## Interface Methods

All backends must implement:

```go
type CacheBackend interface {
    Get(prompt string) (string, bool)
    GetModelInfo(prompt string) (modelName, modelID string, found bool)
    SetWithModel(prompt string, embedding []float32, answer, modelName, modelID string)
    SetPromptWithModel(prompt, answer, modelName, modelID string) error
    GetTopKByEmbedding(embed []float32, k int) []QueryResult
    Flush()
    Stats() (hits, misses uint64, hitRate float64)
}
```

## Error Handling

All storage backends use structured logging for error handling. Errors are logged but not returned to maintain interface compatibility. The logger format is:

```
[backend-name] operation failed for key "key": error details
```

## Testing

Each backend should have corresponding tests. Use the provided test examples as templates.