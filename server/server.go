package server

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/raja-aiml/sematic-cache/core"
)

// Server handles HTTP requests for the cache
type Server struct {
	cache  core.CacheBackend
	router *gin.Engine
}

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
	Prompt    string    `json:"prompt"`     // Required for text search, optional with embedding
	Embedding []float32 `json:"embedding,omitempty"`
	TopK      int       `json:"top_k"`
	Threshold float64   `json:"threshold,omitempty"` // Optional similarity threshold
}

type CacheResponse struct {
	Prompt    string `json:"prompt"`
	Answer    string `json:"answer"`
	ModelName string `json:"model_name,omitempty"`
	ModelID   string `json:"model_id,omitempty"`
	Found     bool   `json:"found"`
}

type SimilarResponse struct {
	Prompt  string             `json:"prompt"`
	Results []core.QueryResult `json:"results"`
}

type StatsResponse struct {
	Hits    uint64  `json:"hits"`
	Misses  uint64  `json:"misses"`
	HitRate float64 `json:"hit_rate"`
}

// New creates a new server instance
func New(cache core.CacheBackend) *Server {
	s := &Server{
		cache:  cache,
		router: gin.Default(),
	}
	s.setupRoutes()
	return s
}

// NewWithMode creates a new server with specific Gin mode
func NewWithMode(cache core.CacheBackend, mode string) *Server {
	gin.SetMode(mode)
	return New(cache)
}

// ServeHTTP implements http.Handler
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.router.ServeHTTP(w, r)
}

// Router returns the Gin router for testing
func (s *Server) Router() *gin.Engine {
	return s.router
}

func (s *Server) setupRoutes() {
	// Health check
	s.router.GET("/health", s.handleHealth)

	// Cache operations
	api := s.router.Group("/api/v1")
	{
		api.POST("/cache/get", s.handleCacheGet)
		api.POST("/cache/set", s.handleCacheSet)
		api.POST("/cache/similar", s.handleCacheSimilar)
		api.GET("/cache/stats", s.handleCacheStats)
		api.POST("/cache/flush", s.handleCacheFlush)
	}

	// Legacy endpoints for backward compatibility
	s.router.POST("/cache/get", s.handleCacheGet)
	s.router.POST("/cache/set", s.handleCacheSet)
	s.router.POST("/cache/similar", s.handleCacheSimilar)
}

func (s *Server) handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":  "healthy",
		"service": "semantic-cache",
	})
}

func (s *Server) handleCacheGet(c *gin.Context) {
	var req GetRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Normalize prompt
	prompt := strings.TrimSpace(req.Prompt)

	// Get from cache
	answer, found := s.cache.Get(prompt)
	if !found {
		c.JSON(http.StatusNotFound, CacheResponse{
			Prompt: prompt,
			Found:  false,
		})
		return
	}

	// Get model info if available
	modelName, modelID, _ := s.cache.GetModelInfo(prompt)

	c.JSON(http.StatusOK, CacheResponse{
		Prompt:    prompt,
		Answer:    answer,
		ModelName: modelName,
		ModelID:   modelID,
		Found:     true,
	})
}

func (s *Server) handleCacheSet(c *gin.Context) {
	var req SetRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Normalize prompt
	prompt := strings.TrimSpace(req.Prompt)

	if len(req.Embedding) > 0 {
		// Set with provided embedding
		s.cache.SetWithModel(prompt, req.Embedding, req.Answer, req.ModelName, req.ModelID)
	} else {
		// Set without embedding (will generate embedding internally if configured)
		if err := s.cache.SetPromptWithModel(prompt, req.Answer, req.ModelName, req.ModelID); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to set cache: %v", err)})
			return
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"prompt": prompt,
	})
}

func (s *Server) handleCacheSimilar(c *gin.Context) {
	var req SimilarRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Validate that either prompt or embedding is provided
	if req.Prompt == "" && len(req.Embedding) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "either 'prompt' text or 'embedding' array must be provided",
		})
		return
	}

	// Default top-k
	if req.TopK <= 0 {
		req.TopK = 5
	}

	var results []core.QueryResult
	var err error

	if len(req.Embedding) > 0 {
		// Use provided embedding (prompt field is optional when embedding is provided)
		results = s.cache.GetTopKByEmbedding(req.Embedding, req.TopK)
	} else {
		// Use text prompt - generate embedding internally
		results, err = s.cache.GetTopKByText(c.Request.Context(), req.Prompt, req.TopK)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": fmt.Sprintf("failed to search by text: %v", err),
			})
			return
		}
	}

	// Filter results by threshold if provided
	if req.Threshold > 0 {
		var filteredResults []core.QueryResult
		for _, r := range results {
			if r.Similarity >= req.Threshold {
				filteredResults = append(filteredResults, r)
			}
		}
		results = filteredResults
	}

	c.JSON(http.StatusOK, SimilarResponse{
		Prompt:  req.Prompt,
		Results: results,
	})
}

func (s *Server) handleCacheStats(c *gin.Context) {
	hits, misses, hitRate := s.cache.Stats()

	c.JSON(http.StatusOK, StatsResponse{
		Hits:    hits,
		Misses:  misses,
		HitRate: hitRate,
	})
}

func (s *Server) handleCacheFlush(c *gin.Context) {
	s.cache.Flush()
	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "cache flushed",
	})
}
