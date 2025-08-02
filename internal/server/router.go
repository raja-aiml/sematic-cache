package server

import (
	"os"

	"github.com/gin-gonic/gin"
	"github.com/raja-aiml/sematic-cache/internal/cache/handlers"
	"github.com/raja-aiml/sematic-cache/internal/config"
	"github.com/raja-aiml/sematic-cache/internal/logger"
	"github.com/raja-aiml/sematic-cache/internal/storage"
)

// SetupRouter configures HTTP routes with health checks
func SetupRouter(cache storage.CacheBackend, cfg *config.EnvConfig) *gin.Engine {
	// Set Gin mode based on environment
	if os.Getenv("GIN_MODE") == "" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()

	// Middleware
	router.Use(logger.RecoveryMiddleware())
	router.Use(logger.GinMiddleware())

	// Health checks
	router.GET(cfg.HealthCheckPath, HealthCheck)
	router.GET(cfg.ReadinessCheckPath, ReadinessCheck(cache))

	// API routes
	api := router.Group("/api/v1")
	{
		api.POST("/get", handlers.HandleGet(cache))
		api.POST("/set", handlers.HandleSet(cache))
		api.POST("/similar", handlers.HandleSimilar(cache))
		api.GET("/stats", handlers.HandleStats(cache))
		api.DELETE("/clear", handlers.HandleClear(cache))
	}

	return router
}
