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

// HandleStats returns cache statistics
func HandleStats(backend storage.CacheBackend) gin.HandlerFunc {
	tracer := otel.Tracer("semantic-cache")
	
	return func(c *gin.Context) {
		// Use the request context which contains trace information from middleware
		ctx := c.Request.Context()
		
		// Create a child span for the stats operation
		_, span := tracer.Start(ctx, "cache.stats",
			trace.WithSpanKind(trace.SpanKindInternal),
			trace.WithAttributes(
				attribute.String("cache.operation", "stats"),
			),
		)
		defer span.End()
		
		hits, misses, hitRate := backend.Stats()
		
		// Add stats attributes to span
		span.SetAttributes(
			attribute.Int64("cache.stats.hits", int64(hits)),
			attribute.Int64("cache.stats.misses", int64(misses)),
			attribute.Float64("cache.stats.hit_rate", hitRate),
		)
		
		c.JSON(http.StatusOK, cache.StatsResponse{
			Hits:    hits,
			Misses:  misses,
			HitRate: hitRate,
		})
	}
}
