// Package server exposes HTTP handlers for cache operations using Gin.

package server

import (
   "log"
   "net/http"

   "github.com/gin-gonic/gin"
   "github.com/raja-aiml/sematic-cache/go/core"
)

// setRequest represents the JSON payload for /set.
type setRequest struct {
   Prompt    string    `json:"prompt"`
   Answer    string    `json:"answer"`
   Embedding []float32 `json:"embedding,omitempty"`
   ModelName string    `json:"modelName,omitempty"`
   ModelID   string    `json:"modelID,omitempty"`
}

// getRequest represents the JSON payload for /get.
type getRequest struct {
   Prompt string `json:"prompt"`
}

// getResponse is the JSON response for /get.
type getResponse struct {
   Answer    string `json:"answer"`
   ModelName string `json:"modelName,omitempty"`
   ModelID   string `json:"modelID,omitempty"`
}

// queryRequest represents the JSON payload for /query.
type queryRequest struct {
   Embedding []float32 `json:"embedding"`
}

// queryResponse is the JSON response for /query.
type queryResponse struct {
   Answer     string  `json:"answer"`
   ModelName  string  `json:"modelName,omitempty"`
   ModelID    string  `json:"modelID,omitempty"`
   Similarity float64 `json:"similarity,omitempty"`
}

// topKRequest represents the JSON payload for /topk.
type topKRequest struct {
   Embedding []float32 `json:"embedding"`
   K         int       `json:"k"`
}

// topKResponseItem is a single result in /topk.
type topKResponseItem struct {
   Prompt     string  `json:"prompt"`
   Answer     string  `json:"answer"`
   Similarity float64 `json:"similarity"`
   ModelName  string  `json:"modelName,omitempty"`
   ModelID    string  `json:"modelID,omitempty"`
}

// New creates a Gin engine with all cache routes configured.
func New(cache core.CacheBackend) *gin.Engine {
   r := gin.Default()

   r.POST("/get", func(c *gin.Context) {
       var req getRequest
       if err := c.ShouldBindJSON(&req); err != nil {
           c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
           return
       }
       ans, ok := cache.Get(req.Prompt)
       if !ok {
           c.Status(http.StatusNotFound)
           return
       }
       modelName, modelID, _ := cache.GetModelInfo(req.Prompt)
       resp := getResponse{Answer: ans, ModelName: modelName, ModelID: modelID}
       c.JSON(http.StatusOK, resp)
   })

   r.POST("/set", func(c *gin.Context) {
       var req setRequest
       if err := c.ShouldBindJSON(&req); err != nil {
           c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
           return
       }
       if len(req.Embedding) > 0 {
           cache.SetWithModel(req.Prompt, req.Embedding, req.Answer, req.ModelName, req.ModelID)
       } else {
           if err := cache.SetPromptWithModel(req.Prompt, req.Answer, req.ModelName, req.ModelID); err != nil {
               log.Printf("embedding failed or disabled: %v, storing without embedding", err)
               cache.SetWithModel(req.Prompt, nil, req.Answer, req.ModelName, req.ModelID)
           }
       }
       c.Status(http.StatusCreated)
   })

   r.POST("/flush", func(c *gin.Context) {
       cache.Flush()
       c.Status(http.StatusOK)
   })

   r.POST("/query", func(c *gin.Context) {
       var req queryRequest
       if err := c.ShouldBindJSON(&req); err != nil {
           c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
           return
       }
       results := cache.GetTopKByEmbedding(req.Embedding, 1)
       if len(results) == 0 {
           c.Status(http.StatusNotFound)
           return
       }
       r0 := results[0]
       resp := queryResponse{Answer: r0.Answer, ModelName: r0.ModelName, ModelID: r0.ModelID, Similarity: r0.Similarity}
       c.JSON(http.StatusOK, resp)
   })

   r.POST("/topk", func(c *gin.Context) {
       var req topKRequest
       if err := c.ShouldBindJSON(&req); err != nil {
           c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
           return
       }
       results := cache.GetTopKByEmbedding(req.Embedding, req.K)
       resp := make([]topKResponseItem, len(results))
       for i, r0 := range results {
           resp[i] = topKResponseItem{Prompt: r0.Prompt, Answer: r0.Answer, Similarity: r0.Similarity, ModelName: r0.ModelName, ModelID: r0.ModelID}
       }
       c.JSON(http.StatusOK, resp)
   })

   r.GET("/health", func(c *gin.Context) {
       c.Status(http.StatusOK)
   })

   r.GET("/metrics", func(c *gin.Context) {
       hits, misses, hitRate := cache.Stats()
       c.JSON(http.StatusOK, gin.H{"hits": hits, "misses": misses, "hitRate": hitRate})
   })

   return r
}