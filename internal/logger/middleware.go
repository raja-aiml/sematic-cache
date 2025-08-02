package logger

import (
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// GinMiddleware returns a gin middleware for structured request logging
func GinMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Generate and set request ID
		requestID := uuid.New().String()
		c.Set("request_id", requestID)
		c.Header("X-Request-ID", requestID)

		// Capture request details
		start := time.Now()
		path := c.Request.URL.Path
		method := c.Request.Method

		// Process request
		c.Next()

		// Log request completion
		logRequest(c, requestID, start, path, method)
	}
}

func logRequest(c *gin.Context, requestID string, start time.Time, path, method string) {
	latency := time.Since(start)
	status := c.Writer.Status()

	fields := Fields{
		"request_id": requestID,
		"method":     method,
		"path":       path,
		"status":     status,
		"latency_ms": latency.Milliseconds(),
		"client_ip":  c.ClientIP(),
	}

	// Add query string if present
	if raw := c.Request.URL.RawQuery; raw != "" {
		fields["query"] = raw
	}

	// Add error if present
	if len(c.Errors) > 0 {
		fields["errors"] = c.Errors.String()
	}

	// Log based on status code
	switch {
	case status >= 500:
		Error("Request failed", fields)
	case status >= 400:
		Warn("Client error", fields)
	default:
		Info("Request completed", fields)
	}
}

// RecoveryMiddleware recovers from panics and logs them
func RecoveryMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		defer func() {
			if err := recover(); err != nil {
				Error("Panic recovered", Fields{
					"request_id": c.GetString("request_id"),
					"panic":      err,
					"path":       c.Request.URL.Path,
					"method":     c.Request.Method,
				})
				c.AbortWithStatus(500)
			}
		}()
		c.Next()
	}
}
