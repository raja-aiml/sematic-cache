# Three-Tier Composite Backend

The composite backend provides a sophisticated multi-tier caching strategy that combines the strengths of different storage backends to optimize performance, cost, and reliability.

## Architecture Overview

```
Request → L1: Memory → L2: Redis → L3: PostgreSQL → OpenAI API
         ↓            ↓           ↓                ↓
         <1μs        100μs       1-10ms           seconds
         Hot cache   Warm cache  Similarity       Generate new
```

## Features

### 1. **Automatic Fallback**
Each tier seamlessly falls back to the next when a cache miss occurs:
- Memory (L1) → Redis (L2) → PostgreSQL (L3) → API

### 2. **Smart Cache Promotion**
When data is found in a lower tier, it's automatically promoted to higher tiers for faster future access.

### 3. **Vector Similarity Search**
Tiers that support vector search (Memory, PostgreSQL) can find semantically similar queries even when there's no exact match.

### 4. **Per-Tier Metrics**
Track hit rates, misses, and performance for each tier independently.

## Configuration

### Basic Configuration

```yaml
cache:
  type: "composite"
  min_similarity: 0.85  # Threshold for similarity matches
  composite:
    promote_on_hit: true  # Enable automatic promotion
    tiers:
      - name: "memory-l1"
        type: "memory"
        priority: 1  # Lower number = higher priority
        capacity: 1000
        eviction_policy: "LRU"
      
      - name: "redis-l2"
        type: "redis"
        priority: 2
        redis:
          addrs:
            - "localhost:6379"
      
      - name: "postgres-l3"
        type: "gorm"
        priority: 3
```

### Advanced Configuration

```yaml
cache:
  type: "composite"
  min_similarity: 0.9
  composite:
    promote_on_hit: true
    tiers:
      # Ultra-fast in-memory cache for hot data
      - name: "hot-cache"
        type: "memory"
        priority: 1
        capacity: 500
        eviction_policy: "LFU"  # Keep frequently used items
      
      # Larger memory cache for warm data
      - name: "warm-cache"
        type: "memory"
        priority: 2
        capacity: 5000
        eviction_policy: "LRU"
      
      # Distributed Redis cache
      - name: "distributed"
        type: "redis"
        priority: 3
        redis:
          addrs:
            - "redis-1:6379"
            - "redis-2:6379"
            - "redis-3:6379"
      
      # Persistent PostgreSQL with vector search
      - name: "persistent"
        type: "gorm"
        priority: 4
```

## Cache Promotion Examples

### Example 1: Exact Match Promotion
```
1. Query: "What is machine learning?"
   - Memory: MISS
   - Redis: HIT ✓
   - Action: Promote to Memory
   - Return answer from Redis

2. Next query: "What is machine learning?"
   - Memory: HIT ✓
   - Return immediately
```

### Example 2: Similarity Search with Caching
```
1. Query: "Explain ML"
   - Memory: MISS (no exact match)
   - Redis: MISS (no exact match)
   - PostgreSQL: Similarity search finds "Explain machine learning" (0.92 similarity)
   - Action: Cache "Explain ML" → answer in Memory + Redis
   - Return answer

2. Next query: "Explain ML"
   - Memory: HIT ✓ (exact match now exists)
   - Return immediately
```

## Performance Characteristics

| Tier | Latency | Capacity | Persistence | Distribution | Vector Search |
|------|---------|----------|-------------|--------------|---------------|
| Memory | <1μs | RAM limited | No | No | Yes |
| Redis | ~100μs | High | Optional | Yes | No |
| PostgreSQL | 1-10ms | Unlimited | Yes | Yes | Yes |

## API Usage

The composite backend implements the standard `CacheBackend` interface:

```go
// No special code needed - works like any other backend
cache, err := storage.NewBackend(cfg, embedFunc)

// Use normally
answer, found := cache.Get("What is AI?")
if !found {
    // Cache miss in all tiers
    answer = callOpenAI("What is AI?")
    cache.SetWithModel("What is AI?", embedding, answer, "gpt-4", "v1")
}
```

## Monitoring

### Overall Statistics
```go
hits, misses, hitRate := cache.Stats()
fmt.Printf("Total: %d hits, %d misses, %.2f%% hit rate\n", 
    hits, misses, hitRate*100)
```

### Per-Tier Statistics
```go
if composite, ok := cache.(*storage.CompositeBackend); ok {
    for _, tier := range composite.GetTiers() {
        fmt.Printf("Tier %s: %d hits, %d misses, %.2f%% hit rate\n",
            tier.Name, tier.Hits, tier.Misses, tier.HitRate*100)
    }
}
```

## Use Cases

### 1. **High-Traffic APIs**
- Memory tier absorbs repeated queries
- Redis handles distributed load
- PostgreSQL provides long-term storage

### 2. **Cost Optimization**
- Reduces OpenAI API calls by 99%+
- Tiered approach minimizes infrastructure costs
- Only promotes frequently accessed data

### 3. **Semantic Search Applications**
- PostgreSQL handles similarity searches
- Popular variations cached in faster tiers
- Automatic learning of query patterns

### 4. **Multi-Region Deployments**
- Local memory cache per instance
- Shared Redis across region
- Global PostgreSQL for persistence

## Best Practices

1. **Tier Sizing**
   - Memory: 10-20% of hot data
   - Redis: 50-80% of warm data
   - PostgreSQL: All historical data

2. **Similarity Threshold**
   - 0.85-0.90 for general queries
   - 0.90-0.95 for technical content
   - 0.95+ for exact matching only

3. **Promotion Strategy**
   - Enable for read-heavy workloads
   - Disable for write-heavy scenarios
   - Monitor promotion overhead

4. **Monitoring**
   - Track per-tier hit rates
   - Alert on low L1/L2 hit rates
   - Monitor promotion latency

## Implementation Details

### Thread Safety
All operations are thread-safe with read/write locks protecting tier access.

### Async Operations
- Promotions happen asynchronously
- Writes to all tiers occur in parallel
- Non-blocking for optimal performance

### Error Handling
- Failures in one tier don't affect others
- Automatic fallback on errors
- Comprehensive error logging

## Future Enhancements

1. **Smart Tier Selection**
   - ML-based tier placement
   - Predictive cache warming
   - Dynamic tier configuration

2. **Advanced Features**
   - Cross-region replication
   - Tier-specific TTLs
   - Custom promotion policies

3. **Integration**
   - Prometheus metrics export
   - Grafana dashboards
   - OpenTelemetry tracing