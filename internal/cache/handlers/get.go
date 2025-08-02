package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/raja-aiml/sematic-cache/internal/cache"
	"github.com/raja-aiml/sematic-cache/internal/logger"
	"github.com/raja-aiml/sematic-cache/internal/storage"
)

// HandleGet retrieves a cached response
func HandleGet(backend storage.CacheBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req cache.GetRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		value, found := backend.Get(req.Prompt)

		// Log cache hit/miss
		logFields := logger.Fields{
			"prompt":     req.Prompt,
			"cache_hit":  found,
			"request_id": c.GetString("request_id"),
		}

		if found {
			logger.Debug("Cache hit", logFields)
		} else {
			logger.Debug("Cache miss", logFields)
		}

		c.JSON(http.StatusOK, cache.CacheResponse{
			Prompt: req.Prompt,
			Answer: value,
			Found:  found,
		})
	}
}
