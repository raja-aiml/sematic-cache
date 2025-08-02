# A2A Protocol Memory System

## Overview

The semantic cache server can be used as a distributed memory system for agents following Google's A2A (Agent-to-Agent) protocol. This provides agents with both short-term and long-term memory capabilities with semantic search.

## Architecture

```
┌─────────────────────┐
│   A2A Agents        │
│  (Google Protocol)  │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│   A2A Memory API    │
│   (REST/gRPC)       │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  Memory Adapter     │
│  - Store/Retrieve   │
│  - Semantic Search  │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
    v             v
┌──────────┐ ┌──────────┐
│Short-Term│ │Long-Term │
│  Memory  │ │  Memory  │
│(In-Memory│ │(Postgres/│
│  + TTL)  │ │  Redis)  │
└──────────┘ └──────────┘
```

## Memory Types

### 1. **Short-Term Memory**
- **Purpose**: Recent conversations and immediate context
- **Storage**: In-memory cache with TTL (5-30 minutes)
- **Use Cases**: Current dialogue, working context, temporary state

### 2. **Long-Term Memory**
- **Purpose**: Persistent knowledge and important information
- **Storage**: PostgreSQL with pgvector or Redis
- **Use Cases**: Facts, learned patterns, agent knowledge base

### 3. **Working Memory**
- **Purpose**: Active task execution and temporary computations
- **Storage**: In-memory with short TTL (1-10 minutes)
- **Use Cases**: Current goals, task progress, intermediate results

### 4. **Episodic Memory**
- **Purpose**: Significant events and experiences
- **Storage**: Long-term persistent with high importance
- **Use Cases**: Past interactions, successful strategies, failures to avoid

### 5. **Semantic Memory**
- **Purpose**: Conceptual knowledge and relationships
- **Storage**: Long-term with vector embeddings
- **Use Cases**: Domain knowledge, concept relationships, learned associations

## API Endpoints

### Memory Operations

#### Store Memory
```http
POST /api/v1/a2a/memory/store
Content-Type: application/json

{
  "agent_id": "agent-123",
  "type": "conversation",
  "content": "User asked about quantum computing",
  "memory_type": "short_term",
  "importance": 0.8,
  "context": {
    "user_id": "user-456",
    "session": "sess-789"
  }
}
```

#### Search Memory
```http
POST /api/v1/a2a/memory/search
Content-Type: application/json

{
  "query": "quantum computing breakthroughs",
  "agent_id": "agent-123",
  "memory_type": "long_term",
  "top_k": 5
}
```

#### Get Conversation History
```http
GET /api/v1/a2a/conversation/{agent_id}/history?limit=20
```

#### Get Relevant Context
```http
POST /api/v1/a2a/context/relevant
Content-Type: application/json

{
  "query": "What do you know about IBM quantum computers?",
  "agent_id": "agent-123",
  "limit": 10
}
```

## Configuration

### Short-Term Memory Configuration
```yaml
short_term:
  type: memory
  capacity: 1000
  ttl: 5m
  eviction_policy: lru
  min_similarity: 0.7
```

### Long-Term Memory Configuration
```yaml
long_term:
  type: gorm  # or redis
  capacity: 100000
  ttl: 720h  # 30 days
  database_url: "postgresql://user:pass@localhost/agent_memory"
```

## Integration Examples

### Python Agent Integration
```python
import requests
import json

class A2AMemory:
    def __init__(self, base_url, agent_id):
        self.base_url = base_url
        self.agent_id = agent_id
    
    def remember(self, content, memory_type="short_term"):
        """Store a memory"""
        response = requests.post(
            f"{self.base_url}/api/v1/a2a/memory/store",
            json={
                "agent_id": self.agent_id,
                "content": content,
                "memory_type": memory_type,
                "importance": 0.8
            }
        )
        return response.json()
    
    def recall(self, query, top_k=5):
        """Search for relevant memories"""
        response = requests.post(
            f"{self.base_url}/api/v1/a2a/memory/search",
            json={
                "query": query,
                "agent_id": self.agent_id,
                "top_k": top_k
            }
        )
        return response.json()["memories"]

# Usage
memory = A2AMemory("http://localhost:8080", "agent-researcher")
memory.remember("The user prefers technical explanations")
relevant = memory.recall("How should I explain this concept?")
```

### Go Agent Integration
```go
package main

import (
    "github.com/raja-aiml/sematic-cache/pkg/agent"
)

func main() {
    // Create memory adapter
    memory := agent.NewA2AMemoryAdapter(
        shortTermCache,
        longTermCache,
        embeddingFunc,
    )
    
    // Store conversation
    msg := &agent.A2AMessage{
        AgentID:    "agent-001",
        Content:    "User asked about machine learning",
        MemoryType: agent.ShortTermMemory,
        Importance: 0.7,
    }
    memory.Store(ctx, msg)
    
    // Search for context
    results, _ := memory.Search(ctx, 
        "machine learning algorithms", 
        "agent-001", 
        agent.LongTermMemory, 
        5,
    )
}
```

## Use Cases

### 1. Multi-Agent Collaboration
Agents share memories through the centralized cache, enabling:
- Knowledge sharing between specialized agents
- Collaborative problem-solving
- Consistent context across agent interactions

### 2. Personalized AI Assistants
- Remember user preferences across sessions
- Maintain conversation context
- Learn from past interactions

### 3. Research Agents
- Store discovered facts in long-term memory
- Retrieve relevant information for new queries
- Build knowledge graphs over time

### 4. Task-Oriented Agents
- Track task progress in working memory
- Remember successful strategies
- Avoid repeating past failures

## Performance Considerations

### Memory Tiers
1. **L1 Cache** (Working Memory): ~1μs latency, 100-1000 items
2. **L2 Cache** (Short-term): ~10μs latency, 1000-10000 items  
3. **L3 Storage** (Long-term): ~1-5ms latency, unlimited items

### Optimization Tips
- Use appropriate memory types for different data
- Set TTLs based on expected usage patterns
- Index frequently searched fields
- Batch memory operations when possible
- Use vector similarity threshold to filter results

## Security

### Agent Authentication
```go
// Middleware for agent authentication
func AgentAuthMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        agentToken := c.GetHeader("X-Agent-Token")
        if !validateAgentToken(agentToken) {
            c.AbortWithStatus(401)
            return
        }
        c.Next()
    }
}
```

### Memory Isolation
- Agents can only access their own memories by default
- Shared memories require explicit permissions
- Sensitive data should be encrypted before storage

## Monitoring

### Metrics to Track
- Memory hit/miss rates per agent
- Query latency by memory type
- Storage capacity utilization
- Agent activity patterns
- Memory importance distribution

### Example Prometheus Metrics
```
# Memory operations
a2a_memory_operations_total{agent_id, operation, memory_type}
a2a_memory_latency_seconds{agent_id, operation, memory_type}

# Cache performance
a2a_cache_hit_rate{agent_id, memory_type}
a2a_cache_size_bytes{memory_type}

# Agent activity
a2a_agent_active_total
a2a_agent_last_activity_timestamp{agent_id}
```

## Future Enhancements

1. **Memory Consolidation**: Automatic promotion of important short-term memories to long-term
2. **Memory Decay**: Gradual reduction of importance for unused memories
3. **Cross-Agent Learning**: Shared knowledge base for agent communities
4. **Memory Compression**: Automatic summarization of similar memories
5. **Causal Memory**: Track cause-effect relationships in episodic memory

## References

- [Google A2A Protocol Specification](https://ai.google/research/a2a)
- [Semantic Cache Documentation](../README.md)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)