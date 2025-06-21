package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/raja-aiml/sematic-cache/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockCache implements core.CacheBackend for testing
type mockCache struct {
	data      map[string]cacheEntry
	hits      uint64
	misses    uint64
	embedFunc core.EmbeddingFunc
}

type cacheEntry struct {
	answer    string
	embedding []float32
	modelName string
	modelID   string
}

func newMockCache() *mockCache {
	return &mockCache{
		data: make(map[string]cacheEntry),
	}
}

func (m *mockCache) Get(prompt string) (string, bool) {
	if entry, ok := m.data[prompt]; ok {
		m.hits++
		return entry.answer, true
	}
	m.misses++
	return "", false
}

func (m *mockCache) GetModelInfo(prompt string) (modelName, modelID string, found bool) {
	if entry, ok := m.data[prompt]; ok {
		return entry.modelName, entry.modelID, true
	}
	return "", "", false
}

func (m *mockCache) SetWithModel(prompt string, embedding []float32, answer, modelName, modelID string) {
	m.data[prompt] = cacheEntry{
		answer:    answer,
		embedding: embedding,
		modelName: modelName,
		modelID:   modelID,
	}
}

func (m *mockCache) SetPromptWithModel(prompt, answer, modelName, modelID string) error {
	// Simulate embedding generation
	m.data[prompt] = cacheEntry{
		answer:    answer,
		embedding: []float32{0.1, 0.2, 0.3},
		modelName: modelName,
		modelID:   modelID,
	}
	return nil
}

func (m *mockCache) GetTopKByEmbedding(embed []float32, k int) []core.QueryResult {
	// Simple mock implementation
	var results []core.QueryResult
	for prompt, entry := range m.data {
		if len(results) < k {
			results = append(results, core.QueryResult{
				Prompt:     prompt,
				Answer:     entry.answer,
				Similarity: 0.9, // Mock similarity score
				ModelName:  entry.modelName,
				ModelID:    entry.modelID,
			})
		}
	}
	return results
}

func (m *mockCache) Flush() {
	m.data = make(map[string]cacheEntry)
	m.hits = 0
	m.misses = 0
}

func (m *mockCache) Stats() (hits, misses uint64, hitRate float64) {
	total := float64(m.hits + m.misses)
	if total > 0 {
		hitRate = float64(m.hits) / total
	}
	return m.hits, m.misses, hitRate
}

func TestServer(t *testing.T) {
	gin.SetMode(gin.TestMode)
	
	t.Run("health check", func(t *testing.T) {
		cache := newMockCache()
		srv := New(cache)
		
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/health", nil)
		srv.Router().ServeHTTP(w, req)
		
		assert.Equal(t, http.StatusOK, w.Code)
		
		var resp map[string]string
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		assert.Equal(t, "healthy", resp["status"])
		assert.Equal(t, "semantic-cache", resp["service"])
	})
	
	t.Run("cache get - found", func(t *testing.T) {
		cache := newMockCache()
		cache.SetWithModel("What is AI?", nil, "Artificial Intelligence", "gpt-4", "gpt-4-0613")
		srv := New(cache)
		
		body := GetRequest{Prompt: "What is AI?"}
		jsonBody, _ := json.Marshal(body)
		
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/api/v1/cache/get", bytes.NewBuffer(jsonBody))
		req.Header.Set("Content-Type", "application/json")
		srv.Router().ServeHTTP(w, req)
		
		assert.Equal(t, http.StatusOK, w.Code)
		
		var resp CacheResponse
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		assert.Equal(t, "What is AI?", resp.Prompt)
		assert.Equal(t, "Artificial Intelligence", resp.Answer)
		assert.Equal(t, "gpt-4", resp.ModelName)
		assert.Equal(t, "gpt-4-0613", resp.ModelID)
		assert.True(t, resp.Found)
	})
	
	t.Run("cache get - not found", func(t *testing.T) {
		cache := newMockCache()
		srv := New(cache)
		
		body := GetRequest{Prompt: "Unknown prompt"}
		jsonBody, _ := json.Marshal(body)
		
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/api/v1/cache/get", bytes.NewBuffer(jsonBody))
		req.Header.Set("Content-Type", "application/json")
		srv.Router().ServeHTTP(w, req)
		
		assert.Equal(t, http.StatusNotFound, w.Code)
		
		var resp CacheResponse
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		assert.False(t, resp.Found)
	})
	
	t.Run("cache set - with embedding", func(t *testing.T) {
		cache := newMockCache()
		srv := New(cache)
		
		body := SetRequest{
			Prompt:    "What is ML?",
			Answer:    "Machine Learning",
			ModelName: "gpt-4",
			Embedding: []float32{0.1, 0.2, 0.3},
		}
		jsonBody, _ := json.Marshal(body)
		
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/api/v1/cache/set", bytes.NewBuffer(jsonBody))
		req.Header.Set("Content-Type", "application/json")
		srv.Router().ServeHTTP(w, req)
		
		assert.Equal(t, http.StatusOK, w.Code)
		
		// Verify it was stored
		answer, found := cache.Get("What is ML?")
		assert.True(t, found)
		assert.Equal(t, "Machine Learning", answer)
	})
	
	t.Run("cache set - without embedding", func(t *testing.T) {
		cache := newMockCache()
		srv := New(cache)
		
		body := SetRequest{
			Prompt:    "What is DL?",
			Answer:    "Deep Learning",
			ModelName: "gpt-4",
		}
		jsonBody, _ := json.Marshal(body)
		
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/api/v1/cache/set", bytes.NewBuffer(jsonBody))
		req.Header.Set("Content-Type", "application/json")
		srv.Router().ServeHTTP(w, req)
		
		assert.Equal(t, http.StatusOK, w.Code)
		
		// Verify it was stored
		answer, found := cache.Get("What is DL?")
		assert.True(t, found)
		assert.Equal(t, "Deep Learning", answer)
	})
	
	t.Run("cache similar - with embedding", func(t *testing.T) {
		cache := newMockCache()
		cache.SetWithModel("What is AI?", []float32{0.1, 0.2, 0.3}, "Artificial Intelligence", "", "")
		srv := New(cache)
		
		body := SimilarRequest{
			Query:     "Tell me about AI",
			Embedding: []float32{0.1, 0.2, 0.3},
			TopK:      2,
		}
		jsonBody, _ := json.Marshal(body)
		
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/api/v1/cache/similar", bytes.NewBuffer(jsonBody))
		req.Header.Set("Content-Type", "application/json")
		srv.Router().ServeHTTP(w, req)
		
		assert.Equal(t, http.StatusOK, w.Code)
		
		var resp SimilarResponse
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		assert.Equal(t, "Tell me about AI", resp.Query)
		assert.Len(t, resp.Results, 1)
		assert.Equal(t, "What is AI?", resp.Results[0].Prompt)
	})
	
	t.Run("cache similar - without embedding", func(t *testing.T) {
		cache := newMockCache()
		srv := New(cache)
		
		body := SimilarRequest{
			Query: "Tell me about AI",
			TopK:  2,
		}
		jsonBody, _ := json.Marshal(body)
		
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/api/v1/cache/similar", bytes.NewBuffer(jsonBody))
		req.Header.Set("Content-Type", "application/json")
		srv.Router().ServeHTTP(w, req)
		
		assert.Equal(t, http.StatusBadRequest, w.Code)
		
		var resp map[string]interface{}
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		assert.Contains(t, resp["error"], "embedding required")
	})
	
	t.Run("cache stats", func(t *testing.T) {
		cache := newMockCache()
		cache.Get("test1") // miss
		cache.SetWithModel("test2", nil, "answer", "", "")
		cache.Get("test2") // hit
		srv := New(cache)
		
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/api/v1/cache/stats", nil)
		srv.Router().ServeHTTP(w, req)
		
		assert.Equal(t, http.StatusOK, w.Code)
		
		var resp StatsResponse
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		assert.Equal(t, uint64(1), resp.Hits)
		assert.Equal(t, uint64(1), resp.Misses)
		assert.Equal(t, 0.5, resp.HitRate)
	})
	
	t.Run("cache flush", func(t *testing.T) {
		cache := newMockCache()
		cache.SetWithModel("test", nil, "answer", "", "")
		srv := New(cache)
		
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/api/v1/cache/flush", nil)
		srv.Router().ServeHTTP(w, req)
		
		assert.Equal(t, http.StatusOK, w.Code)
		
		// Verify cache was flushed
		_, found := cache.Get("test")
		assert.False(t, found)
	})
	
	t.Run("legacy endpoints", func(t *testing.T) {
		cache := newMockCache()
		cache.SetWithModel("test", nil, "answer", "", "")
		srv := New(cache)
		
		// Test legacy /cache/get endpoint
		body := GetRequest{Prompt: "test"}
		jsonBody, _ := json.Marshal(body)
		
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/cache/get", bytes.NewBuffer(jsonBody))
		req.Header.Set("Content-Type", "application/json")
		srv.Router().ServeHTTP(w, req)
		
		assert.Equal(t, http.StatusOK, w.Code)
		
		var resp CacheResponse
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		assert.True(t, resp.Found)
		assert.Equal(t, "answer", resp.Answer)
	})
}