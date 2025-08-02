package server

import (
	"github.com/gin-gonic/gin"
	"github.com/raja-aiml/sematic-cache/internal/logger"
)

// StructuredLoggingMiddleware logs requests in structured format using zap
func StructuredLoggingMiddleware() gin.HandlerFunc {
	return logger.GinMiddleware()
}
