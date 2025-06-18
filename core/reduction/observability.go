package reduction

import (
	"context"
	"fmt"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
)

var (
	tracer = otel.Tracer("github.com/raja-aiml/sematic-cache/core/reduction")
	meter  = otel.Meter("github.com/raja-aiml/sematic-cache/core/reduction")
)

// Prometheus metrics
var (
	// Dimension reduction operations
	reductionOpsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "dimension_reduction_operations_total",
		Help: "Total number of dimension reduction operations",
	}, []string{"operation", "status"})

	reductionDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "dimension_reduction_duration_seconds",
		Help:    "Duration of dimension reduction operations",
		Buckets: prometheus.ExponentialBuckets(0.001, 2, 10), // 1ms to ~1s
	}, []string{"operation"})

	// Quality metrics
	varianceExplainedGauge = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "dimension_reduction_variance_explained",
		Help: "Variance explained by the current PCA model",
	})

	compressionRatioGauge = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "dimension_reduction_compression_ratio",
		Help: "Compression ratio achieved by dimension reduction",
	})

	reconstructionErrorGauge = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "dimension_reduction_reconstruction_error",
		Help: "Average reconstruction error of the current model",
	})

	// Search performance
	searchLatencyHistogram = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "dimension_reduction_search_latency_seconds",
		Help:    "Latency of similarity search operations",
		Buckets: prometheus.ExponentialBuckets(0.0001, 2, 10), // 100µs to ~100ms
	}, []string{"phase", "reduced_dims"})

	candidatesProcessedGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "dimension_reduction_candidates_processed",
		Help: "Number of candidates processed in search",
	}, []string{"phase"})

	// Resource usage
	memoryUsageGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "dimension_reduction_memory_usage_bytes",
		Help: "Memory usage of dimension reduction components",
	}, []string{"component"})

	// Model updates
	modelUpdatesTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "dimension_reduction_model_updates_total",
		Help: "Total number of PCA model updates",
	})

	incrementalUpdatesTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "dimension_reduction_incremental_updates_total",
		Help: "Total number of incremental PCA updates",
	})

	// Quality degradation alerts
	qualityDegradationTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "dimension_reduction_quality_degradation_total",
		Help: "Total number of quality degradation events detected",
	})
)

// OpenTelemetry metrics
var (
	reductionCounter   metric.Int64Counter
	reductionDurationM metric.Float64Histogram
	varianceExplained  metric.Float64Gauge
	compressionRatio   metric.Float64Gauge
)

func init() {
	// Initialize OpenTelemetry metrics
	var err error
	reductionCounter, err = meter.Int64Counter(
		"dimension_reduction_operations",
		metric.WithDescription("Number of dimension reduction operations"),
		metric.WithUnit("1"),
	)
	if err != nil {
		panic(fmt.Errorf("failed to create reduction counter: %w", err))
	}

	reductionDurationM, err = meter.Float64Histogram(
		"dimension_reduction_duration",
		metric.WithDescription("Duration of dimension reduction operations"),
		metric.WithUnit("s"),
	)
	if err != nil {
		panic(fmt.Errorf("failed to create duration histogram: %w", err))
	}

	varianceExplained, err = meter.Float64Gauge(
		"dimension_reduction_variance_explained",
		metric.WithDescription("Variance explained by the PCA model"),
		metric.WithUnit("1"),
	)
	if err != nil {
		panic(fmt.Errorf("failed to create variance gauge: %w", err))
	}

	compressionRatio, err = meter.Float64Gauge(
		"dimension_reduction_compression_ratio",
		metric.WithDescription("Compression ratio achieved"),
		metric.WithUnit("1"),
	)
	if err != nil {
		panic(fmt.Errorf("failed to create compression gauge: %w", err))
	}
}

// ObservableReducer wraps a dimension reducer with observability
type ObservableReducer struct {
	reducer              *OptimizedDimensionReducer
	qualityThreshold     float64
	degradationCallback  func(ctx context.Context, metrics *QualityMetrics)
}

// NewObservableReducer creates a new observable dimension reducer
func NewObservableReducer(config *Config, qualityThreshold float64) (*ObservableReducer, error) {
	reducer, err := NewOptimizedDimensionReducer(config)
	if err != nil {
		return nil, err
	}

	return &ObservableReducer{
		reducer:          reducer,
		qualityThreshold: qualityThreshold,
	}, nil
}

// SetDegradationCallback sets the callback for quality degradation alerts
func (or *ObservableReducer) SetDegradationCallback(callback func(ctx context.Context, metrics *QualityMetrics)) {
	or.degradationCallback = callback
}

// Learn trains the reducer with observability
func (or *ObservableReducer) Learn(ctx context.Context, embeddings [][]float32) error {
	ctx, span := tracer.Start(ctx, "DimensionReducer.Learn",
		trace.WithAttributes(
			attribute.Int("num_samples", len(embeddings)),
			attribute.Int("embedding_dim", len(embeddings[0])),
		))
	defer span.End()

	start := time.Now()
	err := or.reducer.Learn(ctx, embeddings)
	duration := time.Since(start)

	// Record metrics
	status := "success"
	if err != nil {
		status = "error"
		span.RecordError(err)
	}

	reductionOpsTotal.WithLabelValues("learn", status).Inc()
	reductionDuration.WithLabelValues("learn").Observe(duration.Seconds())

	if err == nil {
		// Update quality metrics
		info := or.reducer.GetReductionInfo()
		varianceExplainedGauge.Set(info.VarianceExplained)
		compressionRatioGauge.Set(info.CompressionRatio)

		// OpenTelemetry metrics
		reductionCounter.Add(ctx, 1, metric.WithAttributes(
			attribute.String("operation", "learn"),
			attribute.String("status", status),
		))
		reductionDurationM.Record(ctx, duration.Seconds(), metric.WithAttributes(
			attribute.String("operation", "learn"),
		))
		varianceExplained.Record(ctx, info.VarianceExplained)
		compressionRatio.Record(ctx, info.CompressionRatio)

		modelUpdatesTotal.Inc()
	}

	return err
}

// ReduceForSearch reduces embeddings with observability
func (or *ObservableReducer) ReduceForSearch(ctx context.Context, embedding []float32) ([]float32, error) {
	ctx, span := tracer.Start(ctx, "DimensionReducer.ReduceForSearch",
		trace.WithAttributes(
			attribute.Int("embedding_dim", len(embedding)),
		))
	defer span.End()

	start := time.Now()
	reduced, err := or.reducer.ReduceForSearch(ctx, embedding)
	duration := time.Since(start)

	status := "success"
	if err != nil {
		status = "error"
		span.RecordError(err)
	}

	reductionOpsTotal.WithLabelValues("reduce", status).Inc()
	reductionDuration.WithLabelValues("reduce").Observe(duration.Seconds())

	if err == nil {
		span.SetAttributes(attribute.Int("reduced_dim", len(reduced)))
	}

	return reduced, err
}

// HybridSearch performs search with detailed observability
func (or *ObservableReducer) HybridSearch(
	ctx context.Context,
	queryEmbedding []float32,
	candidates []SearchCandidate,
	topK int,
	similarityFunc func(a, b []float32) float64,
) ([]SearchResult, error) {
	ctx, span := tracer.Start(ctx, "DimensionReducer.HybridSearch",
		trace.WithAttributes(
			attribute.Int("num_candidates", len(candidates)),
			attribute.Int("top_k", topK),
		))
	defer span.End()

	// Perform the search with timing
	start := time.Now()
	results, err := or.reducer.OptimizedHybridSearch(ctx, queryEmbedding, candidates, topK, similarityFunc)
	duration := time.Since(start)
	
	if err != nil {
		span.RecordError(err)
		return nil, err
	}

	searchLatencyHistogram.WithLabelValues("hybrid", "true").Observe(duration.Seconds())
	candidatesProcessedGauge.WithLabelValues("total").Set(float64(len(candidates)))

	// Check for quality degradation
	or.checkQualityDegradation(ctx)

	return results, err
}

// checkQualityDegradation monitors for quality issues
func (or *ObservableReducer) checkQualityDegradation(ctx context.Context) {
	metrics := or.reducer.GetMetrics()
	
	// Check if accuracy has degraded below threshold
	if metrics.AccuracyScore > 0 && metrics.AccuracyScore < or.qualityThreshold {
		qualityDegradationTotal.Inc()
		
		// Create alert span
		_, span := tracer.Start(ctx, "QualityDegradationAlert",
			trace.WithAttributes(
				attribute.Float64("accuracy_score", metrics.AccuracyScore),
				attribute.Float64("threshold", or.qualityThreshold),
				attribute.Float64("variance_explained", metrics.VarianceExplained),
			))
		span.End()

		// Call degradation callback if set
		if or.degradationCallback != nil {
			qm := &QualityMetrics{
				totalQueries:           metrics.TotalQueries,
				reducedDimQueries:      metrics.ReducedDimQueries,
				fullDimReranks:         metrics.FullDimReranks,
			}
			setFloat64Atomic(&qm.avgReductionTimeMs, metrics.AvgReductionTimeMs)
			setFloat64Atomic(&qm.avgRerankTimeMs, metrics.AvgRerankTimeMs)
			setFloat64Atomic(&qm.hitRateBeforeReduction, metrics.HitRateBeforeReduction)
			setFloat64Atomic(&qm.hitRateAfterReduction, metrics.HitRateAfterReduction)
			setFloat64Atomic(&qm.accuracyScore, metrics.AccuracyScore)
			setFloat64Atomic(&qm.varianceExplained, metrics.VarianceExplained)
			setFloat64Atomic(&qm.memorySavedMB, metrics.MemorySavedMB)
			or.degradationCallback(ctx, qm)
		}
	}
}

// UpdateMemoryMetrics updates memory usage metrics
func (or *ObservableReducer) UpdateMemoryMetrics(ctx context.Context) {
	info := or.reducer.GetReductionInfo()
	
	// Estimate memory usage
	pcaMemory := float64(info.OriginalDim*info.ReducedDim*4) / (1024 * 1024) // Components matrix in MB
	memoryUsageGauge.WithLabelValues("pca_components").Set(pcaMemory)
	
	// Get pool stats
	poolStats := GetPoolStats()
	poolMemory := float64(len(poolStats.SlicePoolSizes)*1000*4) / (1024 * 1024) // Rough estimate
	memoryUsageGauge.WithLabelValues("object_pools").Set(poolMemory)
}

// GetMetrics returns current metrics for monitoring
func (or *ObservableReducer) GetMetrics() MetricsSnapshot {
	return or.reducer.GetMetrics()
}

// GetReducer returns the underlying reducer for direct access
func (or *ObservableReducer) GetReducer() *OptimizedDimensionReducer {
	return or.reducer
}

// HealthCheck performs a health check on the dimension reducer
func (or *ObservableReducer) HealthCheck(ctx context.Context) error {
	ctx, span := tracer.Start(ctx, "DimensionReducer.HealthCheck")
	defer span.End()

	// Check if model is learned
	info := or.reducer.GetReductionInfo()
	if !info.IsLearned {
		return fmt.Errorf("PCA model not trained")
	}

	// Check variance explained
	if info.VarianceExplained < 0.8 {
		span.SetAttributes(attribute.Float64("variance_explained", info.VarianceExplained))
		return fmt.Errorf("variance explained too low: %.2f", info.VarianceExplained)
	}

	// Test a reduction operation
	testEmb := make([]float32, info.OriginalDim)
	for i := range testEmb {
		testEmb[i] = float32(i) / float32(info.OriginalDim)
	}

	_, err := or.reducer.ReduceForSearch(ctx, testEmb)
	if err != nil {
		span.RecordError(err)
		return fmt.Errorf("reduction test failed: %w", err)
	}

	return nil
}