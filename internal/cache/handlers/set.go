package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/raja-aiml/sematic-cache/internal/cache"
	"github.com/raja-aiml/sematic-cache/internal/storage"
)

// HandleSet stores a response in cache
func HandleSet(backend storage.CacheBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req cache.SetRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		if err := backend.SetPromptWithModel(req.Prompt, req.Answer, req.ModelName, req.ModelID); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"status": "stored"})
	}
}
