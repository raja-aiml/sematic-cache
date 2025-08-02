package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/raja-aiml/sematic-cache/internal/storage"
)

// HandleClear flushes the cache
func HandleClear(backend storage.CacheBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		backend.Flush()
		c.JSON(http.StatusOK, gin.H{"status": "cleared"})
	}
}
