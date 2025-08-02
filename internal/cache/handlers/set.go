package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/raja-aiml/sematic-cache/internal/cache"
	"github.com/raja-aiml/sematic-cache/internal/logger"
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
			logger.Error("Failed to store in cache", logger.Fields{
				"error":      err.Error(),
				"request_id": c.GetString("request_id"),
			})
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		logger.Debug("Cache entry stored", logger.Fields{
			"prompt":     req.Prompt,
			"model_name": req.ModelName,
			"model_id":   req.ModelID,
			"request_id": c.GetString("request_id"),
		})

		c.JSON(http.StatusOK, gin.H{"status": "stored"})
	}
}
