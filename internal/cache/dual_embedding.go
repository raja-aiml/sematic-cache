package cache

import (
	"context"
	"fmt"
	"time"
)

// DualEmbeddingEntry represents a cache entry with both full and reduced embeddings
type DualEmbeddingEntry struct {
	Prompt           string
	Embedding        []float32
	ReducedEmbedding []float32
	Answer           string
	ModelName        string
	ModelID          string
	Timestamp        time.Time
	LastAccessed     time.Time
	AccessCount      int
	Metadata         map[string]interface{}
}

// DualEmbeddingStorage extends storage backends to support dual embeddings
type DualEmbeddingStorage interface {
	// SetWithDualEmbeddings stores an entry with both full and reduced embeddings
	SetWithDualEmbeddings(ctx context.Context, entry *DualEmbeddingEntry) error

	// GetWithDualEmbeddings retrieves an entry with both embeddings
	GetWithDualEmbeddings(ctx context.Context, prompt string) (*DualEmbeddingEntry, error)

	// SearchByReducedEmbedding performs similarity search using reduced embeddings
	SearchByReducedEmbedding(ctx context.Context, reducedEmbed []float32, topK int, threshold float64) ([]*DualEmbeddingEntry, error)

	// UpdateReducedEmbeddings batch updates reduced embeddings for existing entries
	UpdateReducedEmbeddings(ctx context.Context, updates map[string][]float32) error

	// GetEntriesWithoutReducedEmbeddings returns entries that don't have reduced embeddings
	GetEntriesWithoutReducedEmbeddings(ctx context.Context, limit int) ([]*DualEmbeddingEntry, error)
}

// toDualEmbeddingEntry converts internal entry to DualEmbeddingEntry
func toDualEmbeddingEntry(e *entry) *DualEmbeddingEntry {
	return &DualEmbeddingEntry{
		Prompt:           e.prompt,
		Embedding:        e.embedding,
		ReducedEmbedding: e.reducedEmb,
		Answer:           e.answer,
		ModelName:        e.ModelName,
		ModelID:          e.ModelID,
		Timestamp:        time.Unix(0, e.timestamp),
		LastAccessed:     time.Unix(0, e.lastAccessed),
		AccessCount:      e.accessCount,
		Metadata:         make(map[string]interface{}),
	}
}

// fromDualEmbeddingEntry converts DualEmbeddingEntry to internal entry
func fromDualEmbeddingEntry(de *DualEmbeddingEntry) *entry {
	return &entry{
		prompt:       de.Prompt,
		embedding:    de.Embedding,
		reducedEmb:   de.ReducedEmbedding,
		answer:       de.Answer,
		ModelName:    de.ModelName,
		ModelID:      de.ModelID,
		timestamp:    de.Timestamp.UnixNano(),
		lastAccessed: de.LastAccessed.UnixNano(),
		accessCount:  de.AccessCount,
	}
}

// EnsureBothEmbeddings ensures an entry has both full and reduced embeddings
func (c *Cache) EnsureBothEmbeddings(prompt string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	el, exists := c.entries[prompt]
	if !exists {
		return fmt.Errorf("entry not found: %s", prompt)
	}

	ent := el.Value.(*entry)

	// Ensure we have full embedding
	if len(ent.embedding) == 0 {
		return fmt.Errorf("entry missing full embedding: %s", prompt)
	}

	// Ensure we have reduced embedding if reducer is available
	if c.dimensionReducer != nil && len(ent.reducedEmb) == 0 {
		if c.dimensionReducer.GetReductionInfo().IsLearned {
			ctx := context.Background()
			reduced, err := c.dimensionReducer.ReduceForSearch(ctx, ent.embedding)
			if err != nil {
				return fmt.Errorf("failed to generate reduced embedding: %w", err)
			}
			ent.reducedEmb = reduced
		}
	}

	return nil
}

// GetDualEmbeddingStats returns statistics about dual embedding coverage
func (c *Cache) GetDualEmbeddingStats() DualEmbeddingStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	stats := DualEmbeddingStats{
		TotalEntries: len(c.entries),
	}

	for _, el := range c.entries {
		ent := el.Value.(*entry)

		if len(ent.embedding) > 0 {
			stats.WithFullEmbedding++
		}

		if len(ent.reducedEmb) > 0 {
			stats.WithReducedEmbedding++
		}

		if len(ent.embedding) > 0 && len(ent.reducedEmb) > 0 {
			stats.WithBothEmbeddings++
		}
	}

	if stats.TotalEntries > 0 {
		stats.CoveragePercent = float64(stats.WithBothEmbeddings) / float64(stats.TotalEntries) * 100
	}

	return stats
}

// DualEmbeddingStats contains statistics about dual embedding coverage
type DualEmbeddingStats struct {
	TotalEntries         int
	WithFullEmbedding    int
	WithReducedEmbedding int
	WithBothEmbeddings   int
	CoveragePercent      float64
}
