package observability

import (
	"context"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
)

var (
	tracer trace.Tracer
	meter  metric.Meter

	// Metrics
	cacheHitCounter         metric.Int64Counter
	cacheMissCounter        metric.Int64Counter
	embeddingDuration       metric.Float64Histogram
	similaritySearchDuration metric.Float64Histogram
	cacheOperationDuration  metric.Float64Histogram
	similarityScoreHistogram metric.Float64Histogram
	tierHitCounter          metric.Int64Counter
	evictionCounter         metric.Int64Counter
	embeddingCacheSize      metric.Int64UpDownCounter
)

// InitMetrics initializes custom metrics for semantic cache
func InitMetrics() error {
	tracer = otel.Tracer("semantic-cache")
	meter = otel.Meter("semantic-cache")

	var err error

	// Cache hit/miss counters
	cacheHitCounter, err = meter.Int64Counter(
		"cache.hits",
		metric.WithDescription("Number of cache hits"),
		metric.WithUnit("1"),
	)
	if err != nil {
		return err
	}

	cacheMissCounter, err = meter.Int64Counter(
		"cache.misses",
		metric.WithDescription("Number of cache misses"),
		metric.WithUnit("1"),
	)
	if err != nil {
		return err
	}

	// Embedding generation duration
	embeddingDuration, err = meter.Float64Histogram(
		"embedding.generation.duration",
		metric.WithDescription("Time to generate embeddings"),
		metric.WithUnit("ms"),
	)
	if err != nil {
		return err
	}

	// Similarity search duration
	similaritySearchDuration, err = meter.Float64Histogram(
		"similarity.search.duration",
		metric.WithDescription("Time to perform similarity search"),
		metric.WithUnit("ms"),
	)
	if err != nil {
		return err
	}

	// Cache operation duration
	cacheOperationDuration, err = meter.Float64Histogram(
		"cache.operation.duration",
		metric.WithDescription("Duration of cache operations"),
		metric.WithUnit("ms"),
	)
	if err != nil {
		return err
	}

	// Similarity score distribution
	similarityScoreHistogram, err = meter.Float64Histogram(
		"similarity.score.distribution",
		metric.WithDescription("Distribution of similarity scores"),
		metric.WithUnit("1"),
	)
	if err != nil {
		return err
	}

	// Tier-specific hit counter
	tierHitCounter, err = meter.Int64Counter(
		"cache.tier.hits",
		metric.WithDescription("Number of hits per cache tier"),
		metric.WithUnit("1"),
	)
	if err != nil {
		return err
	}

	// Eviction counter
	evictionCounter, err = meter.Int64Counter(
		"cache.evictions",
		metric.WithDescription("Number of cache evictions"),
		metric.WithUnit("1"),
	)
	if err != nil {
		return err
	}

	// Embedding cache size
	embeddingCacheSize, err = meter.Int64UpDownCounter(
		"embedding.cache.size",
		metric.WithDescription("Current size of embedding cache"),
		metric.WithUnit("1"),
	)
	if err != nil {
		return err
	}

	return nil
}

// RecordCacheHit records a cache hit with tier information
func RecordCacheHit(ctx context.Context, tier string) {
	cacheHitCounter.Add(ctx, 1, metric.WithAttributes(
		attribute.String("tier", tier),
	))
	tierHitCounter.Add(ctx, 1, metric.WithAttributes(
		attribute.String("tier", tier),
	))
}

// RecordCacheMiss records a cache miss
func RecordCacheMiss(ctx context.Context) {
	cacheMissCounter.Add(ctx, 1)
}

// RecordEmbeddingGeneration records embedding generation metrics
func RecordEmbeddingGeneration(ctx context.Context, duration time.Duration, model string, dimensions int) {
	embeddingDuration.Record(ctx, float64(duration.Milliseconds()),
		metric.WithAttributes(
			attribute.String("model", model),
			attribute.Int("dimensions", dimensions),
		))
}

// RecordSimilaritySearch records similarity search metrics
func RecordSimilaritySearch(ctx context.Context, duration time.Duration, resultsCount int, topK int) {
	similaritySearchDuration.Record(ctx, float64(duration.Milliseconds()),
		metric.WithAttributes(
			attribute.Int("results_count", resultsCount),
			attribute.Int("top_k", topK),
		))
}

// RecordSimilarityScore records the distribution of similarity scores
func RecordSimilarityScore(ctx context.Context, score float64) {
	similarityScoreHistogram.Record(ctx, score)
}

// RecordCacheOperation records generic cache operation metrics
func RecordCacheOperation(ctx context.Context, operation string, duration time.Duration, success bool) {
	cacheOperationDuration.Record(ctx, float64(duration.Milliseconds()),
		metric.WithAttributes(
			attribute.String("operation", operation),
			attribute.Bool("success", success),
		))
}

// RecordEviction records cache eviction
func RecordEviction(ctx context.Context, policy string, tier string) {
	evictionCounter.Add(ctx, 1,
		metric.WithAttributes(
			attribute.String("policy", policy),
			attribute.String("tier", tier),
		))
}

// UpdateEmbeddingCacheSize updates the current embedding cache size
func UpdateEmbeddingCacheSize(ctx context.Context, delta int64, tier string) {
	embeddingCacheSize.Add(ctx, delta,
		metric.WithAttributes(
			attribute.String("tier", tier),
		))
}

// StartSpan starts a new span with the given name
func StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
	return tracer.Start(ctx, name, opts...)
}

// TraceEmbeddingGeneration creates a span for embedding generation
func TraceEmbeddingGeneration(ctx context.Context, prompt string, model string) (context.Context, trace.Span) {
	ctx, span := tracer.Start(ctx, "GenerateEmbedding",
		trace.WithAttributes(
			attribute.String("prompt", prompt),
			attribute.String("model", model),
		))
	return ctx, span
}

// TraceSimilaritySearch creates a span for similarity search
func TraceSimilaritySearch(ctx context.Context, prompt string, topK int, threshold float64) (context.Context, trace.Span) {
	ctx, span := tracer.Start(ctx, "SimilaritySearch",
		trace.WithAttributes(
			attribute.String("prompt", prompt),
			attribute.Int("top_k", topK),
			attribute.Float64("threshold", threshold),
		))
	return ctx, span
}

// TraceCacheOperation creates a span for cache operations
func TraceCacheOperation(ctx context.Context, operation string, key string) (context.Context, trace.Span) {
	ctx, span := tracer.Start(ctx, operation,
		trace.WithAttributes(
			attribute.String("cache.key", key),
			attribute.String("cache.operation", operation),
		))
	return ctx, span
}