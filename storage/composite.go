// Package storage provides a three-tier composite backend implementation.
package storage

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/raja-aiml/sematic-cache/core"
)

// TierType represents the type of cache tier
type TierType string

const (
	TierMemory     TierType = "memory"
	TierRedis      TierType = "redis"
	TierPostgreSQL TierType = "postgresql"
)

// TierCapabilities defines what a tier can do
type TierCapabilities struct {
	SupportsVectorSearch bool
	AverageLatencyNs     int64 // Average latency in nanoseconds
	IsPersistent         bool
	IsDistributed        bool
}

// Tier represents a single tier in the composite cache
type Tier struct {
	Type         TierType
	Name         string
	Backend      core.CacheBackend
	Capabilities TierCapabilities
	Priority     int // Lower number = higher priority (checked first)

	// Metrics
	hits   uint64
	misses uint64
}

// CompositeBackend implements a multi-tier caching strategy
type CompositeBackend struct {
	tiers     []*Tier
	embedder  core.EmbeddingFunc
	threshold float64
	logger    *Logger
	mu        sync.RWMutex

	// Promotion settings
	promoteOnHit bool
	promoteTTL   time.Duration
}

// NewCompositeBackend creates a new composite backend with configured tiers
func NewCompositeBackend(tiers []*Tier, embedder core.EmbeddingFunc, threshold float64) *CompositeBackend {
	// Sort tiers by priority
	for i := 0; i < len(tiers)-1; i++ {
		for j := i + 1; j < len(tiers); j++ {
			if tiers[i].Priority > tiers[j].Priority {
				tiers[i], tiers[j] = tiers[j], tiers[i]
			}
		}
	}

	return &CompositeBackend{
		tiers:        tiers,
		embedder:     embedder,
		threshold:    threshold,
		logger:       NewLogger("composite"),
		promoteOnHit: true,
		promoteTTL:   1 * time.Hour,
	}
}

// Get attempts to retrieve a value from the cache tiers
func (c *CompositeBackend) Get(prompt string) (string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// Try exact match in each tier
	for tierIdx, tier := range c.tiers {
		answer, found := tier.Backend.Get(prompt)
		if found {
			atomic.AddUint64(&tier.hits, 1)
			c.logger.LogInfo("get", fmt.Sprintf("hit in tier %s", tier.Name))

			// Promote to higher tiers if enabled
			if c.promoteOnHit && tierIdx > 0 {
				go c.promoteToHigherTiers(prompt, answer, "", "", tierIdx)
			}

			return answer, true
		}
		atomic.AddUint64(&tier.misses, 1)
	}

	// No exact match found, try similarity search
	if c.embedder != nil {
		embedding, err := c.embedder(prompt)
		if err != nil {
			c.logger.LogError("embedding", prompt, err)
			return "", false
		}

		// Try similarity search on capable tiers
		for _, tier := range c.tiers {
			if !tier.Capabilities.SupportsVectorSearch {
				continue
			}

			results := tier.Backend.GetTopKByEmbedding(embedding, 1)
			if len(results) > 0 && results[0].Similarity >= c.threshold {
				atomic.AddUint64(&tier.hits, 1)
				c.logger.LogInfo("get", fmt.Sprintf("similarity hit in tier %s (score: %.3f)", tier.Name, results[0].Similarity))

				// Cache the exact prompt in all tiers
				go c.cacheInAllTiers(prompt, embedding, results[0].Answer, results[0].ModelName, results[0].ModelID)

				return results[0].Answer, true
			}
		}
	}

	c.logger.LogInfo("get", "miss in all tiers")
	return "", false
}

// GetModelInfo retrieves model information from the cache tiers
func (c *CompositeBackend) GetModelInfo(prompt string) (modelName, modelID string, found bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	for _, tier := range c.tiers {
		modelName, modelID, found = tier.Backend.GetModelInfo(prompt)
		if found {
			return modelName, modelID, true
		}
	}

	return "", "", false
}

// SetWithModel stores a value in all tiers
func (c *CompositeBackend) SetWithModel(prompt string, embedding []float32, answer, modelName, modelID string) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var wg sync.WaitGroup
	for _, tier := range c.tiers {
		wg.Add(1)
		go func(t *Tier) {
			defer wg.Done()
			t.Backend.SetWithModel(prompt, embedding, answer, modelName, modelID)
		}(tier)
	}
	wg.Wait()

	c.logger.LogInfo("set", fmt.Sprintf("stored in %d tiers", len(c.tiers)))
}

// SetPromptWithModel stores a prompt without embedding
func (c *CompositeBackend) SetPromptWithModel(prompt, answer, modelName, modelID string) error {
	// Generate embedding if available
	var embedding []float32
	if c.embedder != nil {
		emb, err := c.embedder(prompt)
		if err != nil {
			c.logger.LogError("embedding", prompt, err)
			// Continue without embedding
		} else {
			embedding = emb
		}
	}

	c.SetWithModel(prompt, embedding, answer, modelName, modelID)
	return nil
}

// GetTopKByEmbedding searches for similar embeddings across capable tiers
func (c *CompositeBackend) GetTopKByEmbedding(embed []float32, k int) []core.QueryResult {
	c.mu.RLock()
	defer c.mu.RUnlock()

	resultsMap := make(map[string]core.QueryResult)

	// Collect results from all capable tiers
	for _, tier := range c.tiers {
		if !tier.Capabilities.SupportsVectorSearch {
			continue
		}

		results := tier.Backend.GetTopKByEmbedding(embed, k)
		for _, result := range results {
			// Keep the result with highest similarity
			if existing, ok := resultsMap[result.Prompt]; !ok || result.Similarity > existing.Similarity {
				resultsMap[result.Prompt] = result
			}
		}
	}

	// Convert map to slice and sort by similarity
	results := make([]core.QueryResult, 0, len(resultsMap))
	for _, result := range resultsMap {
		results = append(results, result)
	}

	// Sort by similarity descending
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[i].Similarity < results[j].Similarity {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Return top k
	if len(results) > k {
		results = results[:k]
	}

	return results
}

// Flush clears all tiers
func (c *CompositeBackend) Flush() {
	c.mu.Lock()
	defer c.mu.Unlock()

	var wg sync.WaitGroup
	for _, tier := range c.tiers {
		wg.Add(1)
		go func(t *Tier) {
			defer wg.Done()
			t.Backend.Flush()
		}(tier)
	}
	wg.Wait()

	// Reset metrics
	for _, tier := range c.tiers {
		atomic.StoreUint64(&tier.hits, 0)
		atomic.StoreUint64(&tier.misses, 0)
	}

	c.logger.LogInfo("flush", "flushed all tiers")
}

// Stats returns aggregated statistics from all tiers
func (c *CompositeBackend) Stats() (hits, misses uint64, hitRate float64) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	for _, tier := range c.tiers {
		hits += atomic.LoadUint64(&tier.hits)
		misses += atomic.LoadUint64(&tier.misses)
	}

	total := hits + misses
	if total > 0 {
		hitRate = float64(hits) / float64(total)
	}

	return hits, misses, hitRate
}

// TierStats returns statistics for a specific tier
func (c *CompositeBackend) TierStats(tierName string) (hits, misses uint64, hitRate float64) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	for _, tier := range c.tiers {
		if tier.Name == tierName {
			hits = atomic.LoadUint64(&tier.hits)
			misses = atomic.LoadUint64(&tier.misses)
			total := hits + misses
			if total > 0 {
				hitRate = float64(hits) / float64(total)
			}
			return
		}
	}

	return 0, 0, 0
}

// promoteToHigherTiers promotes a value to tiers with higher priority
func (c *CompositeBackend) promoteToHigherTiers(prompt, answer, modelName, modelID string, foundAtIndex int) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// Promote to all tiers with higher priority (lower index)
	for i := 0; i < foundAtIndex; i++ {
		tier := c.tiers[i]
		tier.Backend.SetWithModel(prompt, nil, answer, modelName, modelID)
		c.logger.LogInfo("promote", fmt.Sprintf("promoted to tier %s", tier.Name))
	}
}

// cacheInAllTiers stores a value in all tiers
func (c *CompositeBackend) cacheInAllTiers(prompt string, embedding []float32, answer, modelName, modelID string) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var wg sync.WaitGroup
	for _, tier := range c.tiers {
		wg.Add(1)
		go func(t *Tier) {
			defer wg.Done()
			t.Backend.SetWithModel(prompt, embedding, answer, modelName, modelID)
		}(tier)
	}
	wg.Wait()
}

// SetPromotionEnabled enables or disables automatic promotion
func (c *CompositeBackend) SetPromotionEnabled(enabled bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.promoteOnHit = enabled
}

// GetTiers returns information about all configured tiers
func (c *CompositeBackend) GetTiers() []TierInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()

	infos := make([]TierInfo, len(c.tiers))
	for i, tier := range c.tiers {
		hits := atomic.LoadUint64(&tier.hits)
		misses := atomic.LoadUint64(&tier.misses)
		total := hits + misses
		hitRate := float64(0)
		if total > 0 {
			hitRate = float64(hits) / float64(total)
		}

		infos[i] = TierInfo{
			Name:         tier.Name,
			Type:         string(tier.Type),
			Priority:     tier.Priority,
			Capabilities: tier.Capabilities,
			Hits:         hits,
			Misses:       misses,
			HitRate:      hitRate,
		}
	}

	return infos
}

// TierInfo provides information about a cache tier
type TierInfo struct {
	Name         string
	Type         string
	Priority     int
	Capabilities TierCapabilities
	Hits         uint64
	Misses       uint64
	HitRate      float64
}
