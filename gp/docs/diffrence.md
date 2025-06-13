# Python vs Go Implementation Differences

This document lists features from the original Python code that are not yet present in the Go rewrite. Items are grouped by subsystem.

## Core Cache Logic
- **Initialization parameters**: Python `Cache.init` allows custom functions for enabling cache, embedding, pre/post processing and chaining multiple caches.
- **API key helpers**: `set_openai_key` and `set_azure_openai_key` configure OpenAI credentials.
- **Go status**: the Go `Cache` now supports `ImportData` and `Flush` but still lacks customizable init hooks.

## Storage Layer
- **Python DataManager**: pluggable layers handle in-memory, file based or vector store backends with eviction policies.
- **PostgreSQL integration**: Python supports pgvector via managers; Go offers a basic `PGStore` with prepared statements and similarity search but lacks higher level DataManager abstractions.

## OpenAI Integration
- **Python adapter**: functions in `gptcache.adapter` intercept OpenAI requests, compute embeddings and cache responses transparently.
- **Client utilities**: asynchronous `_put` and `_get` wrappers in `gptcache.client` for talking to the Python server.
- **Go wrapper**: `openai.Client` exposes only `Complete` and `Embedding` methods and no automatic caching.

## HTTP Server
- **Python FastAPI server**: endpoints for `/put`, `/get`, `/flush`, `/cache_file` download and an OpenAI compatible `/v1/chat/completions` proxy with streaming.
- **Go server**: now supports `/set`, `/get` and `/flush` but lacks the proxy and file download routes.

## Examples and Documentation
- **Python**: numerous example folders demonstrating adapters, embeddings, eviction, integration and session usage. README provides detailed setup and usage instructions.
- **Go**: single `go/examples/simple` program illustrating basic cache operations. README section describes building and running the Go server but lacks advanced tutorials.

These gaps highlight the ongoing migration work required for the Go implementation to reach feature parity with the Python project.
