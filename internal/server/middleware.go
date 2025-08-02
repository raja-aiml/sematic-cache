package server

import (
	"log/slog"
	"time"

	"github.com/gin-gonic/gin"
)

// StructuredLoggingMiddleware logs requests in structured format
func StructuredLoggingMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path
		raw := c.Request.URL.RawQuery

		// Process request
		c.Next()

		// Log request
		latency := time.Since(start)
		if raw != "" {
			path = path + "?" + raw
		}

		slog.Info("request",
			"client_ip", c.ClientIP(),
			"method", c.Request.Method,
			"path", path,
			"status", c.Writer.Status(),
			"latency_ms", latency.Milliseconds(),
			"user_agent", c.Request.UserAgent(),
			"error", c.Errors.String(),
		)
	}
}
