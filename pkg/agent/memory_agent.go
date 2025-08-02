// Package agent implements a 12-Factor Agent for semantic memory
package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/raja-aiml/sematic-cache/internal/cache"
	"github.com/raja-aiml/sematic-cache/internal/storage"
)

// Intent represents the parsed intention from natural language
type Intent string

const (
	IntentStore     Intent = "store"
	IntentRecall    Intent = "recall"
	IntentForget    Intent = "forget"
	IntentSummarize Intent = "summarize"
	IntentUnknown   Intent = "unknown"
)

// MemoryRequest represents a natural language request to the agent
type MemoryRequest struct {
	Query    string                 `json:"query"`
	Context  map[string]interface{} `json:"context,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// MemoryResponse represents the agent's response
type MemoryResponse struct {
	Success bool        `json:"success"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
	Intent  Intent      `json:"intent"`
	Error   string      `json:"error,omitempty"`
}

// SemanticMemoryAgent implements a 12-Factor Agent for memory operations
// Following KISS, YAGNI, DRY, SOLID principles
type SemanticMemoryAgent struct {
	store     *storage.VectorStore
	embedFunc cache.EmbeddingFunc

	// 12-Factor Agent: Stateless reducer pattern
	// All state is derived from inputs, not stored
}

// NewSemanticMemoryAgent creates a new memory agent (KISS: Simple constructor)
func NewSemanticMemoryAgent(store *storage.VectorStore, embedFunc cache.EmbeddingFunc) *SemanticMemoryAgent {
	return &SemanticMemoryAgent{
		store:     store,
		embedFunc: embedFunc,
	}
}

// Process handles natural language input (Factor 1: Natural Language to Tool Calls)
func (a *SemanticMemoryAgent) Process(ctx context.Context, request MemoryRequest) (*MemoryResponse, error) {
	// Parse intent from natural language (KISS: Simple keyword matching for now)
	intent := a.parseIntent(request.Query)

	// Factor 8: Own Your Control Flow - Clear decision path
	switch intent {
	case IntentStore:
		return a.handleStore(ctx, request)
	case IntentRecall:
		return a.handleRecall(ctx, request)
	case IntentForget:
		return a.handleForget(ctx, request)
	case IntentSummarize:
		return a.handleSummarize(ctx, request)
	default:
		return &MemoryResponse{
			Success: false,
			Intent:  IntentUnknown,
			Error:   "Could not understand the request. Try: 'remember...', 'recall...', 'forget...', or 'summarize...'",
		}, nil
	}
}

// parseIntent extracts intent from natural language (YAGNI: Start simple)
func (a *SemanticMemoryAgent) parseIntent(query string) Intent {
	lower := strings.ToLower(query)

	// KISS: Simple keyword matching
	switch {
	case strings.Contains(lower, "remember") ||
		strings.Contains(lower, "store") ||
		strings.Contains(lower, "save"):
		return IntentStore

	case strings.Contains(lower, "recall") ||
		strings.Contains(lower, "find") ||
		strings.Contains(lower, "search") ||
		strings.Contains(lower, "what"):
		return IntentRecall

	case strings.Contains(lower, "forget") ||
		strings.Contains(lower, "delete") ||
		strings.Contains(lower, "remove"):
		return IntentForget

	case strings.Contains(lower, "summarize") ||
		strings.Contains(lower, "summary"):
		return IntentSummarize

	default:
		return IntentUnknown
	}
}

// handleStore processes store requests (Factor 4: Tools are Structured Outputs)
func (a *SemanticMemoryAgent) handleStore(ctx context.Context, request MemoryRequest) (*MemoryResponse, error) {
	// Extract content to store (KISS: Simple extraction)
	content := a.extractContent(request.Query, "remember", "store", "save")
	if content == "" {
		return &MemoryResponse{
			Success: false,
			Intent:  IntentStore,
			Error:   "Could not extract content to store",
		}, nil
	}

	// Store with metadata
	err := a.store.Store(ctx, content, content, "", "")
	if err != nil {
		return &MemoryResponse{
			Success: false,
			Intent:  IntentStore,
			Error:   fmt.Sprintf("Failed to store memory: %v", err),
		}, nil
	}

	return &MemoryResponse{
		Success: true,
		Intent:  IntentStore,
		Message: fmt.Sprintf("Stored: %s", content),
	}, nil
}

// handleRecall processes recall requests
func (a *SemanticMemoryAgent) handleRecall(ctx context.Context, request MemoryRequest) (*MemoryResponse, error) {
	// Extract search query
	query := a.extractContent(request.Query, "recall", "find", "search", "what")
	if query == "" {
		query = request.Query // Use full query if extraction fails
	}

	// Search for similar memories
	results, err := a.store.Search(ctx, query, 5)
	if err != nil {
		return &MemoryResponse{
			Success: false,
			Intent:  IntentRecall,
			Error:   fmt.Sprintf("Failed to recall memories: %v", err),
		}, nil
	}

	if len(results) == 0 {
		return &MemoryResponse{
			Success: true,
			Intent:  IntentRecall,
			Message: "No relevant memories found",
			Data:    []interface{}{},
		}, nil
	}

	// Format results (DRY: Reusable formatting)
	formatted := a.formatResults(results)

	return &MemoryResponse{
		Success: true,
		Intent:  IntentRecall,
		Message: fmt.Sprintf("Found %d relevant memories", len(results)),
		Data:    formatted,
	}, nil
}

// handleForget processes forget requests
func (a *SemanticMemoryAgent) handleForget(ctx context.Context, request MemoryRequest) (*MemoryResponse, error) {
	// Extract what to forget
	target := a.extractContent(request.Query, "forget", "delete", "remove")
	if target == "" {
		return &MemoryResponse{
			Success: false,
			Intent:  IntentForget,
			Error:   "Please specify what to forget",
		}, nil
	}

	// Delete from store
	err := a.store.Delete(ctx, target)
	if err != nil {
		return &MemoryResponse{
			Success: false,
			Intent:  IntentForget,
			Error:   fmt.Sprintf("Failed to forget: %v", err),
		}, nil
	}

	return &MemoryResponse{
		Success: true,
		Intent:  IntentForget,
		Message: fmt.Sprintf("Forgotten: %s", target),
	}, nil
}

// handleSummarize processes summarize requests
func (a *SemanticMemoryAgent) handleSummarize(ctx context.Context, request MemoryRequest) (*MemoryResponse, error) {
	// Get stats from store
	stats, err := a.store.Stats(ctx)
	if err != nil {
		return &MemoryResponse{
			Success: false,
			Intent:  IntentSummarize,
			Error:   fmt.Sprintf("Failed to get summary: %v", err),
		}, nil
	}

	return &MemoryResponse{
		Success: true,
		Intent:  IntentSummarize,
		Message: "Memory summary",
		Data:    stats,
	}, nil
}

// extractContent extracts content after keywords (DRY: Reusable extraction)
func (a *SemanticMemoryAgent) extractContent(query string, keywords ...string) string {
	lower := strings.ToLower(query)

	for _, keyword := range keywords {
		if idx := strings.Index(lower, keyword); idx != -1 {
			// Extract everything after the keyword
			content := query[idx+len(keyword):]
			content = strings.TrimSpace(content)
			// Remove common words
			content = strings.TrimPrefix(content, "that ")
			content = strings.TrimPrefix(content, "about ")
			return strings.TrimSpace(content)
		}
	}

	return ""
}

// formatResults formats search results for response (SOLID: Single Responsibility)
func (a *SemanticMemoryAgent) formatResults(results []cache.QueryResult) []map[string]interface{} {
	formatted := make([]map[string]interface{}, len(results))

	for i, result := range results {
		formatted[i] = map[string]interface{}{
			"content":    result.Answer,
			"similarity": result.Similarity,
			"prompt":     result.Prompt,
		}
	}

	return formatted
}

// Factor 6: Launch/Pause/Resume with Simple APIs
func (a *SemanticMemoryAgent) Start(ctx context.Context) error {
	// Agent is stateless, no startup needed (YAGNI)
	return nil
}

func (a *SemanticMemoryAgent) Stop() error {
	// Clean shutdown if needed
	return a.store.Close()
}

// Factor 12: Stateless Reducer - Pure function approach
func ProcessMemoryRequest(store *storage.VectorStore, embedFunc cache.EmbeddingFunc, request MemoryRequest) (*MemoryResponse, error) {
	agent := NewSemanticMemoryAgent(store, embedFunc)
	return agent.Process(context.Background(), request)
}
