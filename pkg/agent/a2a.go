// Package agent provides A2A (Agent-to-Agent) protocol support with semantic memory
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/raja-aiml/sematic-cache/internal/cache"
)

// MemoryType represents different types of agent memory
type MemoryType string

const (
	ShortTermMemory MemoryType = "short_term"
	LongTermMemory  MemoryType = "long_term"
	WorkingMemory   MemoryType = "working"
	EpisodicMemory  MemoryType = "episodic"
	SemanticMemory  MemoryType = "semantic"
)

// A2AMessage represents a message in the Google A2A protocol
type A2AMessage struct {
	ID          string                 `json:"id"`
	AgentID     string                 `json:"agent_id"`
	Timestamp   time.Time              `json:"timestamp"`
	Type        string                 `json:"type"`
	Content     string                 `json:"content"`
	Context     map[string]interface{} `json:"context,omitempty"`
	Embedding   []float32              `json:"embedding,omitempty"`
	MemoryType  MemoryType             `json:"memory_type"`
	TTL         time.Duration          `json:"ttl,omitempty"`
	Importance  float64                `json:"importance"`
}

// A2AMemoryAdapter provides memory services for A2A protocol agents
type A2AMemoryAdapter struct {
	shortTermCache cache.CacheBackend  // In-memory with TTL
	longTermCache  cache.CacheBackend  // Persistent storage
	embedFunc      cache.EmbeddingFunc // For generating embeddings
}

// NewA2AMemoryAdapter creates a new A2A protocol memory adapter
func NewA2AMemoryAdapter(shortTerm, longTerm cache.CacheBackend, embedFunc cache.EmbeddingFunc) *A2AMemoryAdapter {
	return &A2AMemoryAdapter{
		shortTermCache: shortTerm,
		longTermCache:  longTerm,
		embedFunc:      embedFunc,
	}
}

// Store saves a message to the appropriate memory tier
func (a *A2AMemoryAdapter) Store(ctx context.Context, msg *A2AMessage) error {
	// Generate embedding if not provided
	if len(msg.Embedding) == 0 && a.embedFunc != nil {
		embedding, err := a.embedFunc(msg.Content)
		if err != nil {
			return fmt.Errorf("failed to generate embedding: %w", err)
		}
		msg.Embedding = embedding
	}

	// Serialize message
	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("failed to serialize message: %w", err)
	}

	key := a.generateKey(msg)
	
	// Store based on memory type
	switch msg.MemoryType {
	case ShortTermMemory, WorkingMemory:
		// Store in short-term cache with TTL
		if a.shortTermCache != nil {
			a.shortTermCache.SetWithModel(key, msg.Embedding, string(data), msg.AgentID, msg.ID)
		}
	
	case LongTermMemory, EpisodicMemory, SemanticMemory:
		// Store in long-term persistent cache
		if a.longTermCache != nil {
			a.longTermCache.SetWithModel(key, msg.Embedding, string(data), msg.AgentID, msg.ID)
		}
		
		// Also store in short-term for fast access
		if a.shortTermCache != nil && msg.Importance > 0.7 {
			a.shortTermCache.SetWithModel(key, msg.Embedding, string(data), msg.AgentID, msg.ID)
		}
	
	default:
		// Store in both for unknown types
		if a.shortTermCache != nil {
			a.shortTermCache.SetWithModel(key, msg.Embedding, string(data), msg.AgentID, msg.ID)
		}
		if a.longTermCache != nil {
			a.longTermCache.SetWithModel(key, msg.Embedding, string(data), msg.AgentID, msg.ID)
		}
	}
	
	return nil
}

// Retrieve gets a specific message by key
func (a *A2AMemoryAdapter) Retrieve(ctx context.Context, agentID, messageID string) (*A2AMessage, error) {
	key := fmt.Sprintf("a2a:%s:%s", agentID, messageID)
	
	// Try short-term first (faster)
	if a.shortTermCache != nil {
		if data, found := a.shortTermCache.Get(key); found {
			return a.deserializeMessage(data)
		}
	}
	
	// Fall back to long-term
	if a.longTermCache != nil {
		if data, found := a.longTermCache.Get(key); found {
			// Promote to short-term cache for future access
			if a.shortTermCache != nil {
				// Use SetWithModel for promotion (we don't have the embedding here, so just store the data)
				a.shortTermCache.SetWithModel(key, nil, data, "", "")
			}
			return a.deserializeMessage(data)
		}
	}
	
	return nil, fmt.Errorf("message not found: %s", messageID)
}

// Search performs semantic search across agent memories
func (a *A2AMemoryAdapter) Search(ctx context.Context, query string, agentID string, memoryType MemoryType, topK int) ([]*A2AMessage, error) {
	// Generate query embedding
	embedding, err := a.embedFunc(query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}
	
	// Search in appropriate cache based on memory type
	var searchCache cache.CacheBackend
	switch memoryType {
	case ShortTermMemory, WorkingMemory:
		searchCache = a.shortTermCache
	case LongTermMemory, EpisodicMemory, SemanticMemory:
		searchCache = a.longTermCache
	default:
		// Search both and merge results
		return a.searchBothCaches(embedding, agentID, topK)
	}
	
	if searchCache == nil {
		return nil, fmt.Errorf("cache not available for memory type: %s", memoryType)
	}
	
	// Perform semantic search
	results := searchCache.GetTopKByEmbedding(embedding, topK)
	
	// Convert results to A2A messages
	messages := make([]*A2AMessage, 0, len(results))
	for _, result := range results {
		if msg, err := a.deserializeMessage(result.Answer); err == nil {
			// Filter by agent ID if specified
			if agentID == "" || msg.AgentID == agentID {
				msg.Importance = result.Similarity // Use similarity as importance score
				messages = append(messages, msg)
			}
		}
	}
	
	return messages, nil
}

// Forget removes memories based on criteria
func (a *A2AMemoryAdapter) Forget(ctx context.Context, agentID string, olderThan time.Duration) error {
	// This would need to be implemented based on the cache backend's capabilities
	// For now, we rely on TTL-based expiration
	return fmt.Errorf("explicit forget not yet implemented")
}

// GetConversationHistory retrieves recent conversation history for an agent
func (a *A2AMemoryAdapter) GetConversationHistory(ctx context.Context, agentID string, limit int) ([]*A2AMessage, error) {
	// Search for recent messages from this agent
	query := fmt.Sprintf("agent:%s conversation", agentID)
	return a.Search(ctx, query, agentID, ShortTermMemory, limit)
}

// GetRelevantContext retrieves contextually relevant memories for a query
func (a *A2AMemoryAdapter) GetRelevantContext(ctx context.Context, query string, agentID string, limit int) ([]*A2AMessage, error) {
	// Search across all memory types for relevant context
	shortTermResults, _ := a.Search(ctx, query, agentID, ShortTermMemory, limit/2)
	longTermResults, _ := a.Search(ctx, query, agentID, LongTermMemory, limit/2)
	
	// Merge and deduplicate results
	seen := make(map[string]bool)
	results := make([]*A2AMessage, 0, limit)
	
	for _, msg := range shortTermResults {
		if !seen[msg.ID] {
			seen[msg.ID] = true
			results = append(results, msg)
		}
	}
	
	for _, msg := range longTermResults {
		if !seen[msg.ID] && len(results) < limit {
			seen[msg.ID] = true
			results = append(results, msg)
		}
	}
	
	return results, nil
}

// Helper functions

func (a *A2AMemoryAdapter) generateKey(msg *A2AMessage) string {
	return fmt.Sprintf("a2a:%s:%s", msg.AgentID, msg.ID)
}

func (a *A2AMemoryAdapter) deserializeMessage(data string) (*A2AMessage, error) {
	var msg A2AMessage
	if err := json.Unmarshal([]byte(data), &msg); err != nil {
		return nil, err
	}
	return &msg, nil
}

func (a *A2AMemoryAdapter) searchBothCaches(embedding []float32, agentID string, topK int) ([]*A2AMessage, error) {
	messages := make([]*A2AMessage, 0, topK*2)
	
	// Search short-term
	if a.shortTermCache != nil {
		results := a.shortTermCache.GetTopKByEmbedding(embedding, topK)
		for _, result := range results {
			if msg, err := a.deserializeMessage(result.Answer); err == nil {
				if agentID == "" || msg.AgentID == agentID {
					msg.Importance = result.Similarity
					messages = append(messages, msg)
				}
			}
		}
	}
	
	// Search long-term
	if a.longTermCache != nil {
		results := a.longTermCache.GetTopKByEmbedding(embedding, topK)
		for _, result := range results {
			if msg, err := a.deserializeMessage(result.Answer); err == nil {
				if agentID == "" || msg.AgentID == agentID {
					msg.Importance = result.Similarity * 0.9 // Slightly lower weight for long-term
					messages = append(messages, msg)
				}
			}
		}
	}
	
	// Sort by importance and return top K
	// (In production, you'd want to properly sort and deduplicate)
	if len(messages) > topK {
		messages = messages[:topK]
	}
	
	return messages, nil
}