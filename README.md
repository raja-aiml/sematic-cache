# Semantic Cache - Go Implementation

A high-performance semantic caching system for AI applications built in Go. This system provides intelligent caching of prompts and responses using embedding-based similarity search, supporting multiple storage backends and advanced AI agent capabilities.

## Features

### Core Caching
- **Embedding-based similarity search** using cosine similarity, inner product, or L2 distance
- **Multiple eviction policies**: LRU, FIFO, LFU, and Random Replacement
- **TTL support** with automatic expiration
- **Adaptive thresholding** for dynamic similarity matching
- **ANN index integration** for fast approximate nearest neighbor search
- **Batch operations** for efficient bulk loading
- **Pre/post-processing hooks** for data transformation

### Storage Backends
- **In-memory cache** with configurable capacity and eviction policies
- **PostgreSQL with pgvector** for persistent vector similarity search
- **Redis Cluster** support for distributed caching
- **GORM integration** for advanced PostgreSQL operations

### AI Integration
- **OpenAI API wrapper** with support for:
  - Chat completions (including streaming)
  - Text embeddings
  - Image generation and editing
  - Audio transcription and translation
  - Content moderation
- **Agent system** with context chain management
- **Multi-agent orchestration** with intelligent routing

### Observability
- **OpenTelemetry integration** with Jaeger tracing
- **Cache metrics** (hits, misses, hit rates)
- **Health checks** and monitoring endpoints

## Quick Start

### Installation

```bash
go mod init your-project
go get github.com/raja-aiml/sematic-cache/go
```

### Basic Usage

```go
package main

import (
    "fmt"
    "github.com/raja-aiml/sematic-cache/go/core"
)

func main() {
    // Create an in-memory cache with capacity of 100
    cache := core.NewCache(100)
    
    // Store a prompt-response pair
    cache.Set("What is AI?", nil, "Artificial Intelligence is...")
    
    // Retrieve the cached response
    if answer, found := cache.Get("What is AI?"); found {
        fmt.Println("Cached answer:", answer)
    }
}
```

### With Embeddings and Similarity Search

```go
package main

import (
    "context"
    "fmt"
    "os"
    
    "github.com/raja-aiml/sematic-cache/go/core"
    "github.com/raja-aiml/sematic-cache/go/openai"
)

func main() {
    // Initialize OpenAI client
    client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
    
    // Create cache with embedding function
    cache := core.NewCache(100,
        core.WithEmbeddingFunc(func(prompt string) ([]float32, error) {
            return client.Embedding(context.Background(), prompt)
        }),
        core.WithMinSimilarity(0.8),
    )
    
    // Store with automatic embedding generation
    err := cache.SetPrompt("What is machine learning?", "ML is a subset of AI...")
    if err != nil {
        panic(err)
    }
    
    // Search by embedding similarity
    queryEmbed, _ := client.Embedding(context.Background(), "Explain ML concepts")
    if answer, found := cache.GetByEmbedding(queryEmbed); found {
        fmt.Println("Similar cached answer:", answer)
    }
}
```

## Configuration

### YAML Configuration

Create a `config.yml` file:

```yaml
server:
  address: ":8080"

cache:
  type: "gorm"  # Options: memory, redis, gorm
  capacity: 1000
  eviction_policy: "LRU"  # LRU, FIFO, LFU, RR
  ttl: "1h"
  min_similarity: 0.8

openai:
  api_key: "${OPENAI_API_KEY}"
  base_url: "https://api.openai.com/v1"

# For PostgreSQL backend
database_url: "host=localhost user=postgres dbname=cache sslmode=disable"

# For Redis backend
redis:
  addrs: ["localhost:6379"]
  password: ""
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
export DATABASE_URL="postgres://user:pass@localhost/dbname?sslmode=disable"
export JAEGER_ENDPOINT="http://localhost:14268/api/traces"
```

## Server Mode

### Starting the Server

```bash
# With configuration file
go run cmd/server/main.go -config config.yml

# With environment variables only
go run cmd/server/main.go -address :8080
```

### API Endpoints

#### Store a Cache Entry
```bash
curl -X POST http://localhost:8080/set \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is Go?",
    "answer": "Go is a programming language...",
    "modelName": "gpt-3.5-turbo",
    "modelID": "gpt-3.5-turbo-0613"
  }'
```

#### Retrieve a Cache Entry
```bash
curl -X POST http://localhost:8080/get \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Go?"}'
```

#### Similarity Search
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"embedding": [0.1, 0.2, 0.3, ...]}'
```

#### Top-K Similarity Search
```bash
curl -X POST http://localhost:8080/topk \
  -H "Content-Type: application/json" \
  -d '{"embedding": [0.1, 0.2, 0.3, ...], "k": 5}'
```

#### Health Check
```bash
curl http://localhost:8080/health
```

#### Metrics
```bash
curl http://localhost:8080/metrics
```

## Storage Backends

### PostgreSQL with pgvector

Ensure PostgreSQL has the pgvector extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Configuration:
```go
store, err := storage.NewGormStore(
    "host=localhost user=postgres dbname=cache sslmode=disable",
    time.Hour, // TTL
)
```

### Redis Cluster

```go
redisOpts := &redis.ClusterOptions{
    Addrs:    []string{"localhost:6379"},
    Password: "",
}
client := redis.NewClusterClient(redisOpts)
store := storage.NewRedisStore(client, time.Hour)
```

## Advanced Features

### Custom Similarity Functions

```go
cache := core.NewCache(100,
    core.WithSimilarityFunc(func(a, b []float32) float64 {
        // Custom similarity implementation
        return customSimilarity(a, b)
    }),
)
```

### Adaptive Thresholding

```go
cache := core.NewCache(100,
    core.WithAdaptiveThreshold(func(similarities []float64) float64 {
        // Return dynamic threshold based on current similarities
        return computeMeanThreshold(similarities)
    }),
)
```

### ANN Index Integration

```go
// Implement your ANN index
type MyANNIndex struct { /* ... */ }

func (idx *MyANNIndex) Add(key string, vector []float32) error { /* ... */ }
func (idx *MyANNIndex) Remove(key string) error { /* ... */ }
func (idx *MyANNIndex) Search(vector []float32, k int) ([]string, error) { /* ... */ }

cache := core.NewCache(100, core.WithANNIndex(&MyANNIndex{}))
```

### Pre/Post Processing

```go
cache := core.NewCache(100,
    core.WithPreProcessor(func(prompt string) string {
        return strings.ToLower(strings.TrimSpace(prompt))
    }),
    core.WithPostProcessor(func(answer string) string {
        return fmt.Sprintf("Cached: %s", answer)
    }),
)
```

## AI Agents

The system includes a sophisticated agent framework for managing AI conversations. See [AGENTS.md](AGENTS.md) for detailed documentation on:

- Context chain management
- Multi-agent orchestration
- Expert agent routing
- Conversation state management

## Performance Considerations

### Cache Sizing
- **In-memory**: Suitable for single-instance deployments
- **Redis**: Recommended for distributed applications
- **PostgreSQL**: Best for persistent storage with complex queries

### Embedding Dimensions
- Default support for 1536-dimensional embeddings (OpenAI text-embedding-ada-002)
- Configurable for other embedding models

### Similarity Thresholds
- Start with 0.8 for strict matching
- Lower to 0.6-0.7 for more flexible matching
- Use adaptive thresholding for dynamic adjustment

## Testing

```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run specific package tests
go test ./core
go test ./storage
go test ./server
```

## Examples

Check the `examples/` directory for:
- **Simple usage**: Basic caching operations
- **Advanced usage**: ANN index integration, custom similarity functions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license here]

## Support

For issues and questions:
- Create an issue in the repository
- Check existing documentation
- Review test files for usage examples


# AI Agent System

The semantic cache includes a sophisticated agent framework for managing AI conversations with context awareness, expert specialization, and intelligent routing between multiple agents.

## Overview

The agent system consists of three main components:

1. **ContextChain** - Manages conversation history with automatic eviction
2. **Agent** - Individual AI agents with expert roles and context management
3. **Orchestrator** - Routes conversations between multiple agents based on content

## Core Components

### ContextChain

The `ContextChain` maintains an ordered list of chat messages with automatic size management.

```go
type ContextChain struct {
    messages []openai.ChatMessage
    maxLen   int
}
```

#### Features
- **Automatic eviction**: Removes oldest messages when capacity is exceeded
- **Message ordering**: Maintains chronological conversation flow
- **Memory management**: Configurable maximum length to control memory usage

#### Usage

```go
// Create a context chain with max 10 messages
chain := core.NewContextChain(10)

// Add messages
chain.Add(openai.ChatMessage{Role: "user", Content: "Hello"})
chain.Add(openai.ChatMessage{Role: "assistant", Content: "Hi there!"})

// Get all messages
messages := chain.Messages()

// Clear the chain
chain.Clear()
```

### Agent

An `Agent` represents an individual AI entity with its own expertise and conversation context.

```go
type Agent struct {
    ExpertID string
    Chain    *ContextChain
}
```

#### Features
- **Expert identification**: Each agent has a unique expert ID for specialization
- **Context awareness**: Maintains conversation history across interactions
- **Direct chat integration**: Built-in OpenAI API integration

#### Usage

```go
// Create an agent with expertise in Python programming
agent := core.NewAgent("python-expert", 20) // max 20 messages in context

// Add user message
agent.AddUser("How do I create a list in Python?")

// Chat with OpenAI (requires OpenAI client)
response, err := agent.Chat(ctx, openaiClient, openai.ChatOptions{
    Model: "gpt-3.5-turbo",
})

// The response is automatically added to the agent's context
fmt.Println("Agent response:", response)
```

### Orchestrator

The `Orchestrator` manages multiple agents and routes conversations based on content analysis.

```go
type Orchestrator struct {
    agents map[string]*Agent
    router func(input string) string
    client *openai.Client
    chatOptions openai.ChatOptions
}
```

#### Features
- **Multi-agent management**: Register and manage multiple specialized agents
- **Intelligent routing**: Custom routing functions to select appropriate agents
- **Unified interface**: Single entry point for multi-agent conversations

#### Usage

```go
// Create routing function
router := func(input string) string {
    if strings.Contains(input, "python") || strings.Contains(input, "coding") {
        return "python-expert"
    }
    if strings.Contains(input, "data") || strings.Contains(input, "analysis") {
        return "data-scientist"
    }
    return "general-assistant"
}

// Create orchestrator
orchestrator := core.NewOrchestrator(router, openaiClient, openai.ChatOptions{
    Model: "gpt-3.5-turbo",
})

// Register agents
orchestrator.RegisterAgent("python-expert", core.NewAgent("Expert Python Developer", 15))
orchestrator.RegisterAgent("data-scientist", core.NewAgent("Senior Data Scientist", 15))
orchestrator.RegisterAgent("general-assistant", core.NewAgent("General AI Assistant", 10))

// Route conversation
response, err := orchestrator.Route(ctx, "How do I analyze data with Python pandas?")
// This will route to "data-scientist" agent based on keywords
```

## Advanced Usage Patterns

### Expert Agent Roles

Create specialized agents for different domains:

```go
// Technical Support Agent
techSupport := core.NewAgent(`
You are a technical support specialist for a SaaS platform.
Always ask clarifying questions and provide step-by-step solutions.
Focus on troubleshooting and problem resolution.
`, 25)

// Sales Agent
salesAgent := core.NewAgent(`
You are a sales representative for enterprise software.
Focus on understanding customer needs and presenting solutions.
Always be professional and solution-oriented.
`, 20)

// Product Manager Agent
productManager := core.NewAgent(`
You are a senior product manager with expertise in feature planning.
Analyze requirements and provide strategic recommendations.
Consider user experience and business impact.
`, 30)
```

### Context-Aware Routing

Implement sophisticated routing based on conversation context:

```go
func smartRouter(input string) string {
    input = strings.ToLower(input)
    
    // Technical issues
    if containsAny(input, []string{"error", "bug", "crash", "not working", "issue"}) {
        return "tech-support"
    }
    
    // Sales inquiries
    if containsAny(input, []string{"price", "cost", "purchase", "enterprise", "plan"}) {
        return "sales"
    }
    
    // Product questions
    if containsAny(input, []string{"feature", "roadmap", "capability", "integration"}) {
        return "product"
    }
    
    // Code-related
    if containsAny(input, []string{"code", "api", "sdk", "programming", "development"}) {
        return "developer"
    }
    
    return "general"
}
```

### Multi-Turn Conversations

Manage complex conversations across multiple turns:

```go
func handleCustomerConversation(orchestrator *core.Orchestrator) {
    ctx := context.Background()
    
    conversations := []string{
        "I'm having trouble with the API integration",
        "The authentication keeps failing",
        "I'm using the Python SDK version 2.1",
        "Yes, I have the correct API key",
        "Can you show me a working example?",
    }
    
    for i, message := range conversations {
        fmt.Printf("Turn %d: %s\n", i+1, message)
        
        response, err := orchestrator.Route(ctx, message)
        if err != nil {
            log.Printf("Error: %v", err)
            continue
        }
        
        fmt.Printf("Agent: %s\n\n", response)
    }
}
```

### Agent State Management

Track and manage agent states across sessions:

```go
type AgentManager struct {
    orchestrator *core.Orchestrator
    sessions     map[string]*core.Agent // user_id -> agent
    mu           sync.RWMutex
}

func (am *AgentManager) GetOrCreateSession(userID string, agentType string) *core.Agent {
    am.mu.Lock()
    defer am.mu.Unlock()
    
    if agent, exists := am.sessions[userID]; exists {
        return agent
    }
    
    // Create new agent based on type
    var expertID string
    var maxMessages int
    
    switch agentType {
    case "support":
        expertID = "Technical Support Specialist"
        maxMessages = 50
    case "sales":
        expertID = "Sales Representative"
        maxMessages = 30
    default:
        expertID = "General Assistant"
        maxMessages = 20
    }
    
    agent := core.NewAgent(expertID, maxMessages)
    am.sessions[userID] = agent
    return agent
}

func (am *AgentManager) ClearSession(userID string) {
    am.mu.Lock()
    defer am.mu.Unlock()
    delete(am.sessions, userID)
}
```

## Integration with Semantic Cache

Combine agents with semantic caching for improved performance:

```go
func createCachedAgent(cache core.CacheBackend, openaiClient *openai.Client) *core.Agent {
    agent := core.NewAgent("Cached AI Assistant", 15)
    
    // Custom chat function that checks cache first
    cachedChat := func(ctx context.Context, messages []openai.ChatMessage) (string, error) {
        // Create cache key from conversation
        key := generateCacheKey(messages)
        
        // Check cache first
        if cached, found := cache.Get(key); found {
            return cached, nil
        }
        
        // If not cached, call OpenAI
        response, err := openaiClient.Chat(ctx, messages, openai.ChatOptions{
            Model: "gpt-3.5-turbo",
        })
        if err != nil {
            return "", err
        }
        
        // Cache the response
        cache.SetWithModel(key, nil, response, "gpt-3.5-turbo", "")
        
        return response, nil
    }
    
    return agent
}

func generateCacheKey(messages []openai.ChatMessage) string {
    // Create a deterministic key from the conversation
    var parts []string
    for _, msg := range messages {
        parts = append(parts, fmt.Sprintf("%s:%s", msg.Role, msg.Content))
    }
    return strings.Join(parts, "|")
}
```

## Best Practices

### Context Management

1. **Appropriate context length**: Balance memory usage with conversation quality
   ```go
   // For customer support: longer context for complex issues
   supportAgent := core.NewAgent("support-expert", 50)
   
   // For quick queries: shorter context for efficiency
   quickAgent := core.NewAgent("quick-help", 10)
   ```

2. **Context pruning**: Implement intelligent context management
   ```go
   func pruneContext(agent *core.Agent, maxLength int) {
       messages := agent.Chain.Messages()
       if len(messages) > maxLength {
           // Keep system message + recent messages
           keepMessages := append(
               messages[:1],                    // System message
               messages[len(messages)-maxLength+1:]..., // Recent messages
           )
           agent.Chain.Clear()
           for _, msg := range keepMessages {
               agent.Chain.Add(msg)
           }
       }
   }
   ```

### Routing Strategies

1. **Keyword-based routing**: Simple but effective for most use cases
2. **Embedding-based routing**: Use semantic similarity for better classification
3. **Machine learning routing**: Train models for complex routing decisions

### Error Handling

```go
func robustAgentChat(agent *core.Agent, ctx context.Context, client *openai.Client, input string) (string, error) {
    agent.AddUser(input)
    
    // Retry logic
    for attempts := 0; attempts < 3; attempts++ {
        response, err := agent.Chat(ctx, client, openai.ChatOptions{
            Model:       "gpt-3.5-turbo",
            Temperature: &[]float64{0.7}[0],
            MaxTokens:   &[]int{1000}[0],
        })
        
        if err == nil {
            return response, nil
        }
        
        // Log error and retry
        log.Printf("Attempt %d failed: %v", attempts+1, err)
        time.Sleep(time.Duration(attempts+1) * time.Second)
    }
    
    return "", fmt.Errorf("failed after 3 attempts")
}
```

## Performance Considerations

1. **Memory usage**: Context chains consume memory proportional to their length
2. **API costs**: Each agent chat incurs OpenAI API costs
3. **Concurrent access**: Use appropriate synchronization for multi-user scenarios
4. **Cache integration**: Leverage semantic caching to reduce API calls

## Testing

```go
func TestAgentConversation(t *testing.T) {
    agent := core.NewAgent("test-expert", 5)
    
    // Test context chain
    agent.AddUser("Hello")
    agent.AddAssistant("Hi there!")
    
    messages := agent.Chain.Messages()
    if len(messages) != 2 {
        t.Errorf("Expected 2 messages, got %d", len(messages))
    }
    
    // Test context eviction
    for i := 0; i < 10; i++ {
        agent.AddUser(fmt.Sprintf("Message %d", i))
    }
    
    messages = agent.Chain.Messages()
    if len(messages) > 5 {
        t.Errorf("Expected max 5 messages, got %d", len(messages))
    }
}
```

The agent system provides a powerful foundation for building sophisticated AI applications with context awareness, expert specialization, and intelligent conversation management.