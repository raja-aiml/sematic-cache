package storage

import (
	"context"
	"sync/atomic"

	"github.com/raja-aiml/sematic-cache/internal/cache"
)

// CacheAdapter adapts VectorStore to the legacy CacheBackend interface
type CacheAdapter struct {
	store  *VectorStore
	hits   uint64
	misses uint64
}

// NewCacheAdapter creates a new adapter for backward compatibility
func NewCacheAdapter(store *VectorStore) *CacheAdapter {
	return &CacheAdapter{
		store: store,
	}
}

// Get retrieves an answer by exact prompt match
func (a *CacheAdapter) Get(prompt string) (string, bool) {
	answer, found := a.store.Get(context.Background(), prompt)
	if found {
		atomic.AddUint64(&a.hits, 1)
	} else {
		atomic.AddUint64(&a.misses, 1)
	}
	return answer, found
}

// GetModelInfo returns model metadata for a prompt
func (a *CacheAdapter) GetModelInfo(prompt string) (modelName, modelID string, found bool) {
	// For now, we'll need to do a search to get this info
	// In a production system, you'd want a separate method for this
	_, found = a.store.Get(context.Background(), prompt)
	if !found {
		return "", "", false
	}
	// TODO: Store and retrieve model info properly
	return "", "", found
}

// SetWithModel stores a prompt with embedding and model metadata
func (a *CacheAdapter) SetWithModel(prompt string, embedding []float32, answer, modelName, modelID string) {
	// Ignore errors for backward compatibility
	_ = a.store.StoreWithEmbedding(context.Background(), prompt, embedding, answer, modelName, modelID)
}

// SetPromptWithModel stores a prompt without embedding
func (a *CacheAdapter) SetPromptWithModel(prompt, answer, modelName, modelID string) error {
	return a.store.Store(context.Background(), prompt, answer, modelName, modelID)
}

// GetTopKByEmbedding returns the k most similar entries to an embedding
func (a *CacheAdapter) GetTopKByEmbedding(embed []float32, k int) []cache.QueryResult {
	results, err := a.store.SearchByEmbedding(context.Background(), embed, k)
	if err != nil {
		return nil
	}
	return results
}

// GetTopKByText returns the k most similar entries to a text prompt
func (a *CacheAdapter) GetTopKByText(ctx context.Context, text string, k int) ([]cache.QueryResult, error) {
	return a.store.Search(ctx, text, k)
}

// Flush removes all entries
func (a *CacheAdapter) Flush() {
	_ = a.store.Flush(context.Background())
}

// Stats returns cache statistics
func (a *CacheAdapter) Stats() (hits, misses uint64, hitRate float64) {
	hits = atomic.LoadUint64(&a.hits)
	misses = atomic.LoadUint64(&a.misses)
	total := hits + misses
	if total > 0 {
		hitRate = float64(hits) / float64(total)
	}
	return
}

// CacheBackend defines the interface for backward compatibility
type CacheBackend interface {
	Get(prompt string) (string, bool)
	GetModelInfo(prompt string) (modelName, modelID string, found bool)
	SetWithModel(prompt string, embedding []float32, answer, modelName, modelID string)
	SetPromptWithModel(prompt, answer, modelName, modelID string) error
	GetTopKByEmbedding(embed []float32, k int) []cache.QueryResult
	GetTopKByText(ctx context.Context, text string, k int) ([]cache.QueryResult, error)
	Flush()
	Stats() (hits, misses uint64, hitRate float64)
}
