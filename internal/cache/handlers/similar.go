package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"github.com/raja-aiml/sematic-cache/internal/cache"
	"github.com/raja-aiml/sematic-cache/internal/storage"
)

// HandleSimilar finds similar cached responses
func HandleSimilar(backend storage.CacheBackend) gin.HandlerFunc {
	tracer := otel.Tracer("semantic-cache")
	
	return func(c *gin.Context) {
		// Use the request context which contains trace information from middleware
		ctx := c.Request.Context()
		
		// Create a child span for the similar search operation
		ctx, span := tracer.Start(ctx, "cache.similar",
			trace.WithSpanKind(trace.SpanKindInternal),
			trace.WithAttributes(
				attribute.String("cache.operation", "similar"),
			),
		)
		defer span.End()
		
		var req cache.SimilarRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			span.RecordError(err)
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		if req.TopK == 0 {
			req.TopK = 5
		}

		// Add request attributes to span
		span.SetAttributes(
			attribute.String("cache.prompt", req.Prompt),
			attribute.Int("cache.top_k", req.TopK),
			attribute.Bool("cache.has_embedding", len(req.Embedding) > 0),
		)

		var results []cache.QueryResult
		if len(req.Embedding) > 0 {
			results = backend.GetTopKByEmbedding(req.Embedding, req.TopK)
		} else if req.Prompt != "" {
			var err error
			results, err = backend.GetTopKByText(ctx, req.Prompt, req.TopK)
			if err != nil {
				span.RecordError(err)
				span.SetAttributes(attribute.Bool("cache.error", true))
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
		} else {
			errMsg := "either prompt or embedding required"
			span.SetAttributes(attribute.String("cache.error", errMsg))
			c.JSON(http.StatusBadRequest, gin.H{"error": errMsg})
			return
		}

		// Add result attributes to span
		span.SetAttributes(
			attribute.Int("cache.results_count", len(results)),
		)

		c.JSON(http.StatusOK, cache.SimilarResponse{
			Prompt:  req.Prompt,
			Results: results,
		})
	}
}
