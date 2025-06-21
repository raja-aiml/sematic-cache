# Semantic Cache Server

HTTP server implementation for the semantic cache using Gin framework.

## Features

- RESTful API using Gin framework
- Semantic similarity search
- Cache statistics
- Health checks
- Request validation
- Structured JSON responses

## API Endpoints

### Health Check
```
GET /health
```

Returns server health status.

### Cache Operations

#### Get Cached Response
```
POST /api/v1/cache/get
```

Request:
```json
{
  "prompt": "What is AI?"
}
```

Response (200 OK):
```json
{
  "prompt": "What is AI?",
  "answer": "Artificial Intelligence is...",
  "model_name": "gpt-4",
  "model_id": "gpt-4-0613",
  "found": true
}
```

Response (404 Not Found):
```json
{
  "prompt": "What is AI?",
  "found": false
}
```

#### Set Cache Entry
```
POST /api/v1/cache/set
```

Request:
```json
{
  "prompt": "What is AI?",
  "answer": "Artificial Intelligence is...",
  "model_name": "gpt-4",
  "model_id": "gpt-4-0613",
  "embedding": [0.1, 0.2, ...] // optional
}
```

Response:
```json
{
  "status": "success",
  "prompt": "What is AI?"
}
```

#### Similarity Search
```
POST /api/v1/cache/similar
```

Request:
```json
{
  "query": "Tell me about AI",
  "embedding": [0.1, 0.2, ...], // required
  "top_k": 5
}
```

Response:
```json
{
  "query": "Tell me about AI",
  "results": [
    {
      "prompt": "What is AI?",
      "answer": "Artificial Intelligence is...",
      "score": 0.95
    }
  ]
}
```

#### Cache Statistics
```
GET /api/v1/cache/stats
```

Response:
```json
{
  "hits": 1234,
  "misses": 567,
  "hit_rate": 0.685
}
```

#### Flush Cache
```
POST /api/v1/cache/flush
```

Response:
```json
{
  "status": "success",
  "message": "cache flushed"
}
```

## Usage

```go
import (
    "github.com/raja-aiml/sematic-cache/server"
    "github.com/raja-aiml/sematic-cache/storage"
)

// Create cache backend
cache, err := storage.NewBackend(cfg, embedFunc)
if err != nil {
    log.Fatal(err)
}

// Create server
srv := server.New(cache)

// Or with specific Gin mode
srv := server.NewWithMode(cache, gin.ReleaseMode)

// Use as http.Handler
http.ListenAndServe(":8080", srv)
```

## Legacy Endpoints

For backward compatibility, the following endpoints are also available without the `/api/v1` prefix:
- `POST /cache/get`
- `POST /cache/set`
- `POST /cache/similar`