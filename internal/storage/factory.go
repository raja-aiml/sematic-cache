package storage

import (
	"context"
	"fmt"
	"os"

	"github.com/raja-aiml/sematic-cache/internal/cache"
	"github.com/raja-aiml/sematic-cache/internal/config"
	"github.com/raja-aiml/sematic-cache/internal/storage/pgvector"
)

// VectorStore provides semantic caching with vector similarity search
type VectorStore struct {
	store     *pgvector.Store
	embedFunc cache.EmbeddingFunc
	threshold float64
}

// NewVectorStore creates a new vector store instance
func NewVectorStore(cfg *config.Config, embedFunc cache.EmbeddingFunc) (*VectorStore, error) {
	// Get database connection string
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" && cfg != nil && cfg.Storage.DSN != "" {
		dsn = cfg.Storage.DSN
	}
	if dsn == "" {
		return nil, fmt.Errorf("DATABASE_URL or storage.dsn must be configured")
	}

	// Create pgvector store
	store, err := pgvector.NewStore(dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to create pgvector store: %w", err)
	}

	// Get similarity threshold
	threshold := 0.8 // default
	if cfg != nil && cfg.Storage.SimilarityThreshold > 0 {
		threshold = cfg.Storage.SimilarityThreshold
	}

	return &VectorStore{
		store:     store,
		embedFunc: embedFunc,
		threshold: threshold,
	}, nil
}

// Store saves a prompt with its answer
func (v *VectorStore) Store(ctx context.Context, prompt string, answer string, modelName string, modelID string) error {
	// Generate embedding if function is available
	var embedding []float32
	if v.embedFunc != nil {
		emb, err := v.embedFunc(prompt)
		if err != nil {
			return fmt.Errorf("failed to generate embedding: %w", err)
		}
		embedding = emb
	}

	return v.store.Store(ctx, prompt, embedding, answer, modelName, modelID)
}

// StoreWithEmbedding saves a prompt with a pre-computed embedding
func (v *VectorStore) StoreWithEmbedding(ctx context.Context, prompt string, embedding []float32, answer string, modelName string, modelID string) error {
	return v.store.Store(ctx, prompt, embedding, answer, modelName, modelID)
}

// Get retrieves an answer by exact prompt match
func (v *VectorStore) Get(ctx context.Context, prompt string) (string, bool) {
	return v.store.Get(ctx, prompt)
}

// Search finds similar entries using vector similarity
func (v *VectorStore) Search(ctx context.Context, prompt string, k int) ([]cache.QueryResult, error) {
	// Generate embedding for the prompt
	if v.embedFunc == nil {
		return nil, fmt.Errorf("embedding function not configured")
	}

	embedding, err := v.embedFunc(prompt)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	return v.store.Search(ctx, embedding, k, v.threshold)
}

// SearchByEmbedding finds similar entries using a pre-computed embedding
func (v *VectorStore) SearchByEmbedding(ctx context.Context, embedding []float32, k int) ([]cache.QueryResult, error) {
	return v.store.Search(ctx, embedding, k, v.threshold)
}

// Delete removes an entry by prompt
func (v *VectorStore) Delete(ctx context.Context, prompt string) error {
	return v.store.Delete(ctx, prompt)
}

// Flush removes all entries
func (v *VectorStore) Flush(ctx context.Context) error {
	return v.store.Flush(ctx)
}

// Stats returns store statistics
func (v *VectorStore) Stats(ctx context.Context) (map[string]interface{}, error) {
	return v.store.Stats(ctx)
}

// Close closes the underlying store connection
func (v *VectorStore) Close() error {
	return v.store.Close()
}
