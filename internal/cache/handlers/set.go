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

// HandleSet stores a response in cache
func HandleSet(backend storage.CacheBackend) gin.HandlerFunc {
	tracer := otel.Tracer("semantic-cache")
	
	return func(c *gin.Context) {
		// Use the request context which contains trace information
		ctx := c.Request.Context()
		
		// Create a child span for the cache set operation
		ctx, span := tracer.Start(ctx, "cache.set",
			trace.WithSpanKind(trace.SpanKindInternal),
			trace.WithAttributes(
				attribute.String("cache.operation", "set"),
			),
		)
		defer span.End()
		
		var req cache.SetRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			span.RecordError(err)
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		
		// Add request attributes to span
		span.SetAttributes(
			attribute.String("cache.prompt", req.Prompt),
			attribute.String("cache.model", req.ModelName),
		)

		// TODO: Update backend methods to accept context for proper tracing
		if err := backend.SetPromptWithModel(req.Prompt, req.Answer, req.ModelName, req.ModelID); err != nil {
			span.RecordError(err)
			logger.Error("Failed to store in cache", logger.Fields{
				"error":      err.Error(),
				"request_id": c.GetString("request_id"),
			})
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		
		span.SetAttributes(attribute.Bool("cache.stored", true))

		logger.Debug("Cache entry stored", logger.Fields{
			"prompt":     req.Prompt,
			"model_name": req.ModelName,
			"model_id":   req.ModelID,
			"request_id": c.GetString("request_id"),
		})

		c.JSON(http.StatusOK, gin.H{"status": "stored"})
	}
}
