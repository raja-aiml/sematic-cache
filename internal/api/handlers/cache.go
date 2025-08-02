package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/raja-aiml/sematic-cache/internal/cache"
	"github.com/raja-aiml/sematic-cache/internal/storage"
)

// Request/Response structures
type GetRequest struct {
	Prompt string `json:"prompt" binding:"required"`
}

type SetRequest struct {
	Prompt    string    `json:"prompt" binding:"required"`
	Answer    string    `json:"answer" binding:"required"`
	ModelName string    `json:"model_name,omitempty"`
	ModelID   string    `json:"model_id,omitempty"`
	Embedding []float32 `json:"embedding,omitempty"`
}

type SimilarRequest struct {
	Prompt    string    `json:"prompt"`
	Embedding []float32 `json:"embedding,omitempty"`
	TopK      int       `json:"top_k"`
	Threshold float64   `json:"threshold,omitempty"`
}

type CacheResponse struct {
	Prompt    string `json:"prompt"`
	Answer    string `json:"answer"`
	ModelName string `json:"model_name,omitempty"`
	ModelID   string `json:"model_id,omitempty"`
	Found     bool   `json:"found"`
}

type SimilarResponse struct {
	Prompt  string              `json:"prompt"`
	Results []cache.QueryResult `json:"results"`
}

type StatsResponse struct {
	Hits    uint64  `json:"hits"`
	Misses  uint64  `json:"misses"`
	HitRate float64 `json:"hit_rate"`
}

// CacheHandler handles cache operations - KISS principle
type CacheHandler struct {
	cache storage.CacheBackend
}

// NewCacheHandler creates a new cache handler
func NewCacheHandler(cache storage.CacheBackend) *CacheHandler {
	return &CacheHandler{cache: cache}
}

// HandleGet retrieves a cached response
func (h *CacheHandler) HandleGet(c *gin.Context) {
	var req GetRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	value, found := h.cache.Get(req.Prompt)
	c.JSON(http.StatusOK, CacheResponse{
		Prompt: req.Prompt,
		Answer: value,
		Found:  found,
	})
}

// HandleSet stores a response in cache
func (h *CacheHandler) HandleSet(c *gin.Context) {
	var req SetRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Store with model information
	if err := h.cache.SetPromptWithModel(req.Prompt, req.Answer, req.ModelName, req.ModelID); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "stored"})
}

// HandleSimilar finds similar cached responses
func (h *CacheHandler) HandleSimilar(c *gin.Context) {
	var req SimilarRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Default top_k if not specified
	if req.TopK == 0 {
		req.TopK = 5
	}

	// Search by embedding or text
	var results []cache.QueryResult
	if len(req.Embedding) > 0 {
		// Search by embedding
		results = h.cache.GetTopKByEmbedding(req.Embedding, req.TopK)
	} else if req.Prompt != "" {
		// Search by text
		var err error
		results, err = h.cache.GetTopKByText(c.Request.Context(), req.Prompt, req.TopK)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
	} else {
		c.JSON(http.StatusBadRequest, gin.H{"error": "either prompt or embedding required"})
		return
	}

	c.JSON(http.StatusOK, SimilarResponse{
		Prompt:  req.Prompt,
		Results: results,
	})
}

// HandleStats returns cache statistics
func (h *CacheHandler) HandleStats(c *gin.Context) {
	hits, misses, hitRate := h.cache.Stats()
	c.JSON(http.StatusOK, StatsResponse{
		Hits:    hits,
		Misses:  misses,
		HitRate: hitRate,
	})
}

// HandleClear flushes the cache
func (h *CacheHandler) HandleClear(c *gin.Context) {
	h.cache.Flush()
	c.JSON(http.StatusOK, gin.H{"status": "cleared"})
}
