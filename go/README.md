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

## Testing
Run all unit tests:
```bash
go test ./...
```

## Example
```bash
go run examples/simple/main.go
```

## License
MIT