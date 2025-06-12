# Agent Development Guide

This repository is transitioning from the existing Python implementation to a high-performance Go
version. Only OpenAI and PostgreSQL using the `pgvector` extension need to be supported in the
initial migration.

## Migration Plan
1. **Assess Current Code** – Review the Python modules in `gptcache` and `gptcache_server` to
   understand the caching API, data manager interfaces and server behaviour.
2. **Create Go Module** – Add a new directory `go/` at the repository root and initialize it with
   `go mod init`. Organize packages as `core`, `storage`, `openai`, and `server`.
3. **Core Cache Logic** – Implement a `Cache` struct that mirrors the behaviour of the Python
   `Cache` class (init, get, set). Use goroutines and sync primitives to maximise throughput.
4. **PostgreSQL Storage** – Provide a storage layer using `database/sql` with connection pooling
   and prepared statements. Use the `pgvector` extension to store and index embeddings for
   similarity search alongside prompts and responses.
5. **OpenAI Integration** – Implement a minimal client wrapper for the OpenAI API. Only the
   endpoints required by the current project need to be exposed.
6. **HTTP Server** – Implement a REST (or gRPC) server exposing cache operations. The API should be
   compatible with the existing Python server so other languages can talk to it.
7. **Testing** – Write unit tests for all Go packages and run them with `go test ./...`. Keep the
   Python tests runnable with `pytest` until the migration finishes.
8. **Documentation** – Document build steps, running the server and examples in the README. Update
   docs when new Go modules appear.

## Contribution Guidelines
- Place all Go code under the `go/` directory.
- Format Go files with `gofmt -w` and run `go vet ./...` and `go test ./...` before committing.
- When working on Python code, run `pytest` as well.
- Provide clear commit messages referencing the module or feature being ported.

Follow this plan for future work on the Go migration.
