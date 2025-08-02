package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/raja-aiml/sematic-cache/internal/cache"
	"github.com/raja-aiml/sematic-cache/internal/storage"
)

// HandleStats returns cache statistics
func HandleStats(backend storage.CacheBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		hits, misses, hitRate := backend.Stats()
		c.JSON(http.StatusOK, cache.StatsResponse{
			Hits:    hits,
			Misses:  misses,
			HitRate: hitRate,
		})
	}
}
