package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"github.com/raja-aiml/sematic-cache/internal/cache"
	"github.com/raja-aiml/sematic-cache/internal/logger"
	"github.com/raja-aiml/sematic-cache/internal/storage"
)

// HandleGet retrieves a cached response
func HandleGet(backend storage.CacheBackend) gin.HandlerFunc {
	tracer := otel.Tracer("semantic-cache")
	
	return func(c *gin.Context) {
		// Use the request context which contains trace information from middleware
		ctx := c.Request.Context()
		
		// Create a child span for the cache get operation
		ctx, span := tracer.Start(ctx, "cache.get",
			trace.WithSpanKind(trace.SpanKindInternal),
			trace.WithAttributes(
				attribute.String("cache.operation", "get"),
			),
		)
		defer span.End()
		
		var req cache.GetRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			span.RecordError(err)
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// Add request attributes to span
		span.SetAttributes(
			attribute.String("cache.prompt", req.Prompt),
		)

		value, found := backend.Get(req.Prompt)

		// Add result attributes to span
		span.SetAttributes(
			attribute.Bool("cache.hit", found),
		)

		// Log cache hit/miss
		logFields := logger.Fields{
			"prompt":     req.Prompt,
			"cache_hit":  found,
			"request_id": c.GetString("request_id"),
		}

		if found {
			logger.Debug("Cache hit", logFields)
			span.SetAttributes(attribute.String("cache.answer_preview", truncateString(value, 50)))
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

// truncateString truncates a string to the specified length
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
