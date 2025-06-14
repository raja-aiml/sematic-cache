# Sematic Cache (Go)

This repository provides:
  - An in-memory LRU cache with embedding similarity lookup (`core` package)
  - OpenAI API client wrapper for completions and embeddings (`openai` package)
  - HTTP server exposing cache endpoints (`server` package and `cmd/server`)
  - PostgreSQL-backed store with `pgvector` support and prepared statements (`storage` package)
  - Observability with OpenTelemetry and Jaeger exporter (`observability` package)

## Prerequisites
- Go 1.23 or newer
- (Optional) PostgreSQL server with `pgvector` extension for `storage.PGStore`

## Build & Run
1. Clone the repo
2. Build the server:
   ```bash
   cd cmd/server
   go build
   ```
3. Run the server (default listens on `:8080`):
   ```bash
   export JAEGER_ENDPOINT=http://localhost:14268/api/traces
   ./server --address :8080
   ```

## Usage
- **Set** value: POST `/set` with JSON `{ "prompt": "key", "answer": "value" }`
- **Get** value: POST `/get` with JSON `{ "prompt": "key" }`, returns `{ "answer": "value" }`
- **Flush** cache: POST `/flush`

## Configuration

The in-memory cache in the `core` package can be customized via options passed to `core.NewCache`:

- **Eviction policy** (`WithEvictionPolicy`): LRU (default), FIFO, LFU (least-frequently-used), or RR (random replacement).
- **Time-to-Live** (`WithTTL`): expire entries older than the given duration.
- **Similarity metric**:
  - `WithInnerProduct()` to use raw dot-product similarity
  - `WithL2Similarity()` to use an L2-based score (1/(1+distance))
  - `WithSimilarityFunc(fn func(a, b []float32) float64)` for a custom function
- **Minimum similarity threshold** (`WithMinSimilarity`): ignore candidates below this score.
- **Adaptive thresholding** (`WithAdaptiveThreshold`): compute a dynamic cutoff from the top-K similarities.
- **Approximate nearest neighbor** (`WithANNIndex`): plug in an `ANNIndex` (e.g. HNSW/LSH) for sub-linear similarity search.
- **Cache enable filter** (`WithCacheEnable`): skip caching for prompts that do not satisfy the filter.

Example:
```go
import (
  "time"
  "github.com/raja-aiml/sematic-cache/go/core"
)

// hnsw is a user-provided ANN index implementing core.ANNIndex
cache := core.NewCache(100,
  core.WithEvictionPolicy(core.PolicyLFU),
  core.WithTTL(10*time.Minute),
  core.WithInnerProduct(),
  core.WithMinSimilarity(0.5),
  core.WithAdaptiveThreshold(func(sims []float64) float64 {
    // keep only sims >= mean(sims)
    var sum float64
    for _, v := range sims { sum += v }
    return sum / float64(len(sims))
  }),
  core.WithANNIndex(hnsw),
)

// Get stats at runtime:
hits, misses, hitRate := cache.Stats()
fmt.Printf("hits=%d, misses=%d, rate=%.2f\n", hits, misses, hitRate)
```

## Testing
Run all unit tests:
```bash
go test ./...
```

## Example
```bash
go run examples/simple/main.go
```

## Advanced Example
This example demonstrates inner-product similarity, a simple ANNIndex plugin, and adaptive thresholding.
```bash
go run examples/advanced/main.go
```

## License
MIT