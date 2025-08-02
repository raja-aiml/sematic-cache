package server

import (
	"context"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/raja-aiml/sematic-cache/internal/storage"
)

// HealthCheck returns basic health status
func HealthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "healthy",
		"time":   time.Now().Unix(),
	})
}

// ReadinessCheck verifies all dependencies are ready
func ReadinessCheck(cache storage.CacheBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(c.Request.Context(), 3*time.Second)
		defer cancel()

		// Try a simple operation to verify database connection
		if _, err := cache.GetTopKByText(ctx, "health_check", 1); err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status": "not_ready",
				"error":  "database not accessible",
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"status": "ready",
			"time":   time.Now().Unix(),
		})
	}
}
