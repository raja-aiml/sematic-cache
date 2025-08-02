package storage

import (
	"context"
	"testing"
	"time"

	"github.com/raja-aiml/sematic-cache/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockBackend is a test implementation of CacheBackend
type MockBackend struct {
	data         map[string]entry
	getCount     int
	setCount     int
	vectorSearch bool
}

type entry struct {
	answer    string
	embedding []float32
	modelName string
	modelID   string
}

func NewMockBackend(vectorSearch bool) *MockBackend {
	return &MockBackend{
		data:         make(map[string]entry),
		vectorSearch: vectorSearch,
	}
}

func (m *MockBackend) Get(prompt string) (string, bool) {
	m.getCount++
	if e, ok := m.data[prompt]; ok {
		return e.answer, true
	}
	return "", false
}

func (m *MockBackend) GetModelInfo(prompt string) (string, string, bool) {
	if e, ok := m.data[prompt]; ok {
		return e.modelName, e.modelID, true
	}
	return "", "", false
}

func (m *MockBackend) SetWithModel(prompt string, embedding []float32, answer, modelName, modelID string) {
	m.setCount++
	m.data[prompt] = entry{
		answer:    answer,
		embedding: embedding,
		modelName: modelName,
		modelID:   modelID,
	}
}

func (m *MockBackend) SetPromptWithModel(prompt, answer, modelName, modelID string) error {
	m.SetWithModel(prompt, nil, answer, modelName, modelID)
	return nil
}

func (m *MockBackend) GetTopKByEmbedding(embed []float32, k int) []core.QueryResult {
	if !m.vectorSearch {
		return nil
	}

	// Simple mock: return first k items with similarity 0.9
	results := make([]core.QueryResult, 0, k)
	count := 0
	for prompt, e := range m.data {
		if count >= k {
			break
		}
		results = append(results, core.QueryResult{
			Prompt:     prompt,
			Answer:     e.answer,
			Similarity: 0.9,
			ModelName:  e.modelName,
			ModelID:    e.modelID,
		})
		count++
	}
	return results
}

func (m *MockBackend) Flush() {
	m.data = make(map[string]entry)
}

func (m *MockBackend) Stats() (uint64, uint64, float64) {
	return 0, 0, 0
}

func (m *MockBackend) GetTopKByText(ctx context.Context, text string, k int) ([]core.QueryResult, error) {
	// Mock implementation - return empty results
	return []core.QueryResult{}, nil
}

func TestCompositeBackend_BasicOperations(t *testing.T) {
	// Create three tiers
	memoryBackend := NewMockBackend(true)
	redisBackend := NewMockBackend(false)
	pgBackend := NewMockBackend(true)

	tiers := []*Tier{
		{
			Type:     TierMemory,
			Name:     "memory",
			Backend:  memoryBackend,
			Priority: 1,
			Capabilities: TierCapabilities{
				SupportsVectorSearch: true,
			},
		},
		{
			Type:     TierRedis,
			Name:     "redis",
			Backend:  redisBackend,
			Priority: 2,
			Capabilities: TierCapabilities{
				SupportsVectorSearch: false,
			},
		},
		{
			Type:     TierPostgreSQL,
			Name:     "postgresql",
			Backend:  pgBackend,
			Priority: 3,
			Capabilities: TierCapabilities{
				SupportsVectorSearch: true,
			},
		},
	}

	embedder := func(s string) ([]float32, error) {
		return []float32{0.1, 0.2, 0.3}, nil
	}

	composite := NewCompositeBackend(tiers, embedder, 0.8)

	t.Run("SetAndGet", func(t *testing.T) {
		// Set a value
		composite.SetWithModel("test-prompt", nil, "test-answer", "gpt-3.5", "v1")

		// Should be in all tiers
		assert.Equal(t, 1, memoryBackend.setCount)
		assert.Equal(t, 1, redisBackend.setCount)
		assert.Equal(t, 1, pgBackend.setCount)

		// Get should hit memory tier first
		answer, found := composite.Get("test-prompt")
		assert.True(t, found)
		assert.Equal(t, "test-answer", answer)
		assert.Equal(t, 1, memoryBackend.getCount)
		assert.Equal(t, 0, redisBackend.getCount) // Should not reach Redis
		assert.Equal(t, 0, pgBackend.getCount)    // Should not reach PostgreSQL
	})

	t.Run("CachePromotion", func(t *testing.T) {
		// Clear memory tier
		memoryBackend.Flush()

		// Get should hit Redis and promote to memory
		answer, found := composite.Get("test-prompt")
		assert.True(t, found)
		assert.Equal(t, "test-answer", answer)

		// Verify it was promoted
		time.Sleep(100 * time.Millisecond) // Allow async promotion
		memAnswer, memFound := memoryBackend.Get("test-prompt")
		assert.True(t, memFound)
		assert.Equal(t, "test-answer", memAnswer)
	})

	t.Run("SimilaritySearch", func(t *testing.T) {
		// Clear all tiers
		composite.Flush()

		// Add some data to PostgreSQL only
		pgBackend.SetWithModel("machine learning", []float32{0.1, 0.2, 0.3}, "ML answer", "", "")

		// Query with similar prompt
		answer, found := composite.Get("what is ML")
		assert.True(t, found)
		assert.Equal(t, "ML answer", answer)

		// Should be cached in all tiers now
		time.Sleep(100 * time.Millisecond)
		memAnswer, memFound := memoryBackend.Get("what is ML")
		assert.True(t, memFound)
		assert.Equal(t, "ML answer", memAnswer)
	})
}

func TestCompositeBackend_GetTopKByEmbedding(t *testing.T) {
	memoryBackend := NewMockBackend(true)
	redisBackend := NewMockBackend(false)
	pgBackend := NewMockBackend(true)

	// Add different data to each backend
	memoryBackend.SetWithModel("memory-item", nil, "memory-answer", "", "")
	pgBackend.SetWithModel("pg-item", nil, "pg-answer", "", "")

	tiers := []*Tier{
		{Name: "memory", Backend: memoryBackend, Priority: 1, Capabilities: TierCapabilities{SupportsVectorSearch: true}},
		{Name: "redis", Backend: redisBackend, Priority: 2, Capabilities: TierCapabilities{SupportsVectorSearch: false}},
		{Name: "postgresql", Backend: pgBackend, Priority: 3, Capabilities: TierCapabilities{SupportsVectorSearch: true}},
	}

	composite := NewCompositeBackend(tiers, nil, 0.8)

	// Should get results from both vector-capable tiers
	results := composite.GetTopKByEmbedding([]float32{0.1, 0.2}, 5)
	assert.Len(t, results, 2)

	// Verify results contain items from both backends
	prompts := make(map[string]bool)
	for _, r := range results {
		prompts[r.Prompt] = true
	}
	assert.True(t, prompts["memory-item"])
	assert.True(t, prompts["pg-item"])
}

func TestCompositeBackend_Stats(t *testing.T) {
	memoryBackend := NewMockBackend(true)

	tiers := []*Tier{
		{Name: "memory", Backend: memoryBackend, Priority: 1, Capabilities: TierCapabilities{SupportsVectorSearch: true}},
	}

	composite := NewCompositeBackend(tiers, nil, 0.8)

	// Perform some operations
	composite.SetWithModel("test", nil, "answer", "", "")
	composite.Get("test")    // hit
	composite.Get("missing") // miss

	// Check aggregated stats
	hits, misses, hitRate := composite.Stats()
	assert.Equal(t, uint64(1), hits)
	assert.Equal(t, uint64(1), misses)
	assert.Equal(t, 0.5, hitRate)

	// Check tier-specific stats
	tierHits, tierMisses, tierHitRate := composite.TierStats("memory")
	assert.Equal(t, uint64(1), tierHits)
	assert.Equal(t, uint64(1), tierMisses)
	assert.Equal(t, 0.5, tierHitRate)
}

func TestCompositeBackend_DisablePromotion(t *testing.T) {
	memoryBackend := NewMockBackend(true)
	redisBackend := NewMockBackend(false)

	tiers := []*Tier{
		{Name: "memory", Backend: memoryBackend, Priority: 1, Capabilities: TierCapabilities{SupportsVectorSearch: true}},
		{Name: "redis", Backend: redisBackend, Priority: 2, Capabilities: TierCapabilities{SupportsVectorSearch: false}},
	}

	composite := NewCompositeBackend(tiers, nil, 0.8)
	composite.SetPromotionEnabled(false)

	// Add to Redis only
	redisBackend.SetWithModel("test", nil, "answer", "", "")

	// Get should not promote
	answer, found := composite.Get("test")
	assert.True(t, found)
	assert.Equal(t, "answer", answer)

	// Verify it was NOT promoted
	time.Sleep(100 * time.Millisecond)
	_, memFound := memoryBackend.Get("test")
	assert.False(t, memFound)
}

func TestCompositeBackend_TierInfo(t *testing.T) {
	tiers := []*Tier{
		{
			Name:     "memory",
			Type:     TierMemory,
			Backend:  NewMockBackend(true),
			Priority: 1,
			Capabilities: TierCapabilities{
				SupportsVectorSearch: true,
				AverageLatencyNs:     1000,
			},
		},
	}

	composite := NewCompositeBackend(tiers, nil, 0.8)

	infos := composite.GetTiers()
	require.Len(t, infos, 1)

	assert.Equal(t, "memory", infos[0].Name)
	assert.Equal(t, "memory", infos[0].Type)
	assert.Equal(t, 1, infos[0].Priority)
	assert.True(t, infos[0].Capabilities.SupportsVectorSearch)
	assert.Equal(t, int64(1000), infos[0].Capabilities.AverageLatencyNs)
}
