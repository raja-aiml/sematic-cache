Semantic Cache Production Readiness: Recommendations

This document provides an exhaustive roadmap for making the sematic-cache project production-ready. Each section outlines specific gaps, recommended changes, and detailed instructions to implement them.

⸻

1. Architecture & Code Organization

✅ Current Strengths
	•	Clear modular structure: core, server, storage, config, observability, openai.
	•	Supports multiple cache backends and ANN plugins.

🔧 Improvements

1.1 Extract ANN Plugins
	•	Action: Move NaiveANNIndex to a new package:

mkdir -p core/ann
mv examples/advanced/naive_ann.go core/ann/naive.go


	•	Benefit: Formalizes ANN plugin interface, makes testing and benchmarking easier.

1.2 Rename server Package
	•	Action: Rename the server package to httpapi for clarity:

mv server httpapi
sed -i '' 's|/server|/httpapi|g' $(find . -name '*.go')



⸻

2. Embedding & Storage

✅ Current Strengths
	•	Pluggable embedding function via OpenAI.
	•	Support for Redis and Postgres with pgvector.

🔧 Improvements

2.1 Validate Embedding Vector Dimensions
	•	Action: In SetWithModel, ensure embedding dimension == expected (e.g., 1536):

if len(embedding) != 1536 {
  return fmt.Errorf("invalid embedding length: got %d, want 1536", len(embedding))
}



2.2 Batch Redis Operations
	•	Action: For batch sets, use MSET with JSON-encoded values.
	•	Benefit: Reduces latency by avoiding per-entry round trips.

2.3 Optional Vector DB Backends
	•	Plan: Abstract ANNIndex further to support Qdrant, Milvus, Weaviate.
	•	Next Step: Define a VectorDBIndex interface in core/ann/vector_db.go.

⸻

3. Observability

✅ Current Strengths
	•	Integrated with OpenTelemetry + Jaeger.

🔧 Improvements

3.1 Add Prometheus Metrics
	•	Action:
	1.	Install Prometheus client:

go get github.com/prometheus/client_golang/prometheus


	2.	Create observability/metrics.go to track hits, misses, duration:

var CacheHits = prometheus.NewCounter(...)
http.Handle("/metrics", promhttp.Handler())



3.2 Add Logging Middleware
	•	Action: Create middleware/logging.go:

func Logging(next http.Handler) http.Handler {
  return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    start := time.Now()
    next.ServeHTTP(w, r)
    log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
  })
}


	•	Integrate in httpapi.New() using s.mux.Handle("/get", Logging(s.handleGet))

3.3 Add Readiness Probes
	•	Action:
	•	Add /ready for DB check
	•	Add /live that always returns 200 OK

⸻

4. Security & Input Validation

🚧 Critical Issues
	•	/flush is publicly accessible. Inputs are not validated strictly.

🔧 Improvements

4.1 Protect /flush Endpoint
	•	Action: Require an API key via header:

if r.Header.Get("X-Admin-Token") != os.Getenv("ADMIN_TOKEN") {
  http.Error(w, "unauthorized", http.StatusUnauthorized)
  return
}



4.2 Validate JSON Input
	•	Action:
	•	Validate that prompt, answer are non-empty
	•	Validate embedding length if provided
	•	Use go-playground/validator or manual checks

4.3 Structured Logging
	•	Action: Replace log.Printf with zap:

logger, _ := zap.NewProduction()
logger.Info("server started", zap.String("addr", addr))



⸻

5. CLI & Config

✅ Current Strengths
	•	CLI with --config and --address flags.

🔧 Improvements

5.1 Add Log-Level Control
	•	Action: Add a --log-level flag and set it in the logger.
	•	Values: debug, info, warn, error

5.2 Add Storage/Policy CLI Overrides
	•	Flags: --storage=memory|redis|gorm, --eviction-policy=LFU|FIFO|RR|LRU
	•	Parse in main.go and override loaded config if provided

⸻

6. Testing

✅ Current Strengths
	•	Unit tests with mocks and coverage of all major modules.

🔧 Improvements

6.1 Add Integration Tests
	•	Action:
	•	Start server on test port
	•	Use http.Post to /set and /get
	•	Validate persistence and correct response

6.2 Add Fuzzing
	•	Action: Add to core/cache_test.go:

func FuzzCacheGet(f *testing.F) {
  f.Fuzz(func(t *testing.T, input string) {
    c := NewCache(5)
    c.Set(input, []float32{1}, "val")
    _, _ = c.Get(input)
  })
}



6.3 Add Benchmarks
	•	Action: Create core/cache_bench_test.go:

func BenchmarkGet(b *testing.B) {
  c := NewCache(1000)
  for i := 0; i < b.N; i++ {
    c.Get("key")
  }
}



⸻

7. Deployment

🔧 Improvements

7.1 Dockerize the Server
	•	Dockerfile:

FROM golang:1.22 as builder
WORKDIR /app
COPY . .
RUN go build -o server ./cmd/server

FROM gcr.io/distroless/static
COPY --from=builder /app/server /server
ENTRYPOINT ["/server"]



7.2 Add Helm/K8s Support
	•	Files: charts/sematic-cache/{Chart.yaml, values.yaml, templates/}
	•	Add livenessProbe, readinessProbe, envFrom, and resources

7.3 Add GitHub Actions
	•	.github/workflows/test.yaml:

steps:
  - uses: actions/checkout@v3
  - name: Test
    run: go test ./... -v
  - name: Lint
    run: golangci-lint run



⸻

8. Documentation

🚧 Missing
	•	No README.md, no API docs, no Swagger/OpenAPI.

🔧 Improvements

8.1 Create README.md
	•	Include:
	•	What the project does
	•	How to run it
	•	Example curl commands for /set, /get, /query
	•	Configuration format

8.2 Add Swagger/OpenAPI
	•	Use swaggo/swag annotations in httpapi/server.go
	•	Generate with:

swag init --parseDependency --parseInternal


	•	Serve under /docs using http.FileServer

⸻

9. Optional Features
	•	Add import/export CLI for JSONL or CSV
	•	Add support for versioned APIs (/v1/get, /v2/query)
	•	Add rate limiting middleware
	•	Add CLI tool to warm cache from prebuilt embeddings

⸻

✅ Suggested Execution Order
	1.	Protect /flush with API key check
	2.	Add Prometheus metrics + logging middleware
	3.	Add structured logger using zap
	4.	Add /ready and /live endpoints
	5.	Validate and sanitize all inputs
	6.	Add CLI flags for log level and storage override
	7.	Add Swagger docs and basic README
	8.	Dockerize and Helmify the deployment stack
	9.	Add E2E and fuzz tests
	10.	Optional: pluggable vector DB + ANN support