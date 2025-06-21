package reduction

import (
	"context"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// OptimizedDimensionReducer manages dimension reduction with performance optimizations
type OptimizedDimensionReducer struct {
	mu            sync.RWMutex
	reducer       Reducer // Use generic Reducer interface
	config        ReducerConfig
	metrics       *QualityMetrics
	originalDim   int
	reducedDim    int
	varianceRatio []float64
	isLearned     bool
	topKSelector  *TopKSelector
	useObjectPool bool
}

// NewOptimizedDimensionReducer creates a new optimized dimension reducer
// Deprecated: Use NewOptimizedDimensionReducerWithFactory instead
func NewOptimizedDimensionReducer(config *Config) (*OptimizedDimensionReducer, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	// Convert Config to ReducerConfig
	// Use TargetDim if TargetDimensions is not set (for backward compatibility)
	targetDim := config.TargetDimensions
	if targetDim == 0 {
		targetDim = config.TargetDim
	}
	varianceRetained := config.MinVarianceRetained
	if varianceRetained == 0 {
		varianceRetained = config.VarianceThreshold
	}

	reducerConfig := ReducerConfig{
		OutputDimensions: targetDim,
		VarianceRetained: varianceRetained,
	}

	// Create PCA with Config type
	pcaConfig := &Config{
		TargetDim:         reducerConfig.OutputDimensions,
		VarianceThreshold: reducerConfig.VarianceRetained,
	}

	return &OptimizedDimensionReducer{
		config:        reducerConfig,
		metrics:       &QualityMetrics{},
		reducer:       NewPCAGonumReducer(pcaConfig),
		topKSelector:  NewTopKSelector(10), // Default top-K size
		useObjectPool: true,
	}, nil
}

// NewOptimizedDimensionReducerWithFactory creates an optimized reducer using the factory pattern
func NewOptimizedDimensionReducerWithFactory(config DimensionReducerConfig) (*OptimizedDimensionReducer, error) {
	factory := NewReducerFactory()

	// Create the appropriate reducer based on configuration
	var reducer Reducer
	var err error

	// For optimized reducer, prefer Gonum implementations when available
	if config.Type == PCAReducerType {
		reducer, err = factory.CreateReducer(PCAGonumReducerType, config.ReducerConfig)
	} else {
		reducer, err = factory.CreateReducer(config.Type, config.ReducerConfig)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to create reducer: %w", err)
	}

	return &OptimizedDimensionReducer{
		config:        config.ReducerConfig,
		metrics:       &QualityMetrics{},
		reducer:       reducer,
		topKSelector:  NewTopKSelector(10), // Default top-K size
		useObjectPool: true,
	}, nil
}

// SetTopK updates the top-K value for selection
func (dr *OptimizedDimensionReducer) SetTopK(k int) {
	dr.mu.Lock()
	dr.topKSelector = NewTopKSelector(k)
	dr.mu.Unlock()
}

// EnableObjectPooling enables or disables object pooling
func (dr *OptimizedDimensionReducer) EnableObjectPooling(enable bool) {
	dr.mu.Lock()
	dr.useObjectPool = enable
	dr.mu.Unlock()
}

// Learn trains the reducer on sample embeddings using optimized Gonum implementation
func (dr *OptimizedDimensionReducer) Learn(ctx context.Context, embeddings [][]float32) error {
	dr.mu.Lock()
	defer dr.mu.Unlock()

	if len(embeddings) == 0 {
		return fmt.Errorf("no embeddings provided")
	}

	startTime := time.Now()

	// Train reducer with optimization
	if err := dr.reducer.Fit(ctx, embeddings); err != nil {
		return fmt.Errorf("failed to fit reducer: %w", err)
	}

	dr.originalDim = dr.reducer.OriginalDim()
	dr.reducedDim = dr.reducer.ReducedDim()
	dr.varianceRatio = dr.reducer.ExplainedVarianceRatio()
	dr.isLearned = true

	// Calculate quality metrics
	dr.updateLearnMetrics(embeddings, time.Since(startTime))

	return nil
}

// ReduceForSearch reduces embeddings for fast initial search
func (dr *OptimizedDimensionReducer) ReduceForSearch(ctx context.Context, embedding []float32) ([]float32, error) {
	dr.mu.RLock()
	if !dr.isLearned {
		dr.mu.RUnlock()
		return nil, fmt.Errorf("reducer not learned yet")
	}
	dr.mu.RUnlock()

	startTime := time.Now()

	// Use object pool if enabled
	var reduced []float32
	if dr.useObjectPool {
		reduced = GetEmbedding(dr.reducedDim)
		defer PutEmbedding(reduced)
	}

	reducedBatch, err := dr.reducer.Transform(ctx, [][]float32{embedding})
	if err != nil {
		return nil, err
	}

	dr.updateReductionMetrics(time.Since(startTime))

	// Copy result if using pool
	if dr.useObjectPool {
		result := make([]float32, len(reducedBatch[0]))
		copy(result, reducedBatch[0])
		return result, nil
	}

	return reducedBatch[0], nil
}

// OptimizedHybridSearch performs fast search with heap-based top-K selection
func (dr *OptimizedDimensionReducer) OptimizedHybridSearch(
	ctx context.Context,
	queryEmbedding []float32,
	candidates []SearchCandidate,
	topK int,
	similarityFunc func(a, b []float32) float64,
) ([]SearchResult, error) {

	if !dr.isLearned {
		// Fallback to full dimension search
		return dr.optimizedFullDimensionSearch(queryEmbedding, candidates, topK, similarityFunc), nil
	}

	// Phase 1: Fast search with reduced dimensions
	reducedQuery, err := dr.ReduceForSearch(ctx, queryEmbedding)
	if err != nil {
		return nil, fmt.Errorf("failed to reduce query: %w", err)
	}

	startPhase1 := time.Now()

	// Get 3x topK candidates using reduced dimensions for better coverage
	phase1Candidates := dr.optimizedSearchReduced(reducedQuery, candidates, topK*3, similarityFunc)

	phase1Time := time.Since(startPhase1)

	// Phase 2: Re-rank top candidates with full dimensions
	startPhase2 := time.Now()
	finalResults := dr.optimizedRerankWithFullDims(queryEmbedding, phase1Candidates, topK, similarityFunc)
	phase2Time := time.Since(startPhase2)

	// Update metrics
	dr.updateSearchMetrics(phase1Time, phase2Time, len(phase1Candidates), topK)

	return finalResults, nil
}

// optimizedSearchReduced performs similarity search using reduced dimensions with heap
func (dr *OptimizedDimensionReducer) optimizedSearchReduced(
	queryReduced []float32,
	candidates []SearchCandidate,
	topK int,
	similarityFunc func(a, b []float32) float64,
) []SearchCandidate {

	// Use object pool for scored candidates
	var scored []scoredCandidate
	if dr.useObjectPool {
		// Pre-allocate with expected size
		scored = make([]scoredCandidate, 0, len(candidates))
	}

	// Score all candidates
	for _, candidate := range candidates {
		if len(candidate.ReducedEmbedding) == 0 {
			continue
		}

		sim := similarityFunc(queryReduced, candidate.ReducedEmbedding)

		if dr.useObjectPool {
			sc := GetScoredCandidate()
			sc.candidate = candidate
			sc.similarity = sim
			scored = append(scored, *sc)
		} else {
			scored = append(scored, scoredCandidate{
				candidate:  candidate,
				similarity: sim,
			})
		}
	}

	// Use heap-based selection for O(n log k) complexity
	selector := NewTopKSelector(topK)
	results := selector.SelectTopK(scored)

	// Clean up pooled objects
	if dr.useObjectPool {
		for i := range scored {
			PutScoredCandidate(&scored[i])
		}
	}

	return results
}

// optimizedRerankWithFullDims re-ranks candidates using full dimension embeddings with heap
func (dr *OptimizedDimensionReducer) optimizedRerankWithFullDims(
	queryFull []float32,
	candidates []SearchCandidate,
	topK int,
	similarityFunc func(a, b []float32) float64,
) []SearchResult {

	// Use object pool for scored results
	var scored []scoredResult
	if dr.useObjectPool {
		scored = make([]scoredResult, 0, len(candidates))
	}

	for _, candidate := range candidates {
		sim := similarityFunc(queryFull, candidate.Embedding)

		if dr.useObjectPool {
			sr := GetScoredResult()
			sr.result = SearchResult{
				Candidate:       candidate,
				Similarity:      sim,
				UsedReducedDims: false,
			}
			sr.similarity = sim
			scored = append(scored, *sr)
		} else {
			scored = append(scored, scoredResult{
				result: SearchResult{
					Candidate:       candidate,
					Similarity:      sim,
					UsedReducedDims: false,
				},
				similarity: sim,
			})
		}
	}

	// Use heap-based selection
	selector := NewTopKSelector(topK)
	results := selector.SelectTopKResults(scored)

	// Clean up pooled objects
	if dr.useObjectPool {
		for i := range scored {
			PutScoredResult(&scored[i])
		}
	}

	return results
}

// optimizedFullDimensionSearch performs search using only full dimensions with heap
func (dr *OptimizedDimensionReducer) optimizedFullDimensionSearch(
	queryFull []float32,
	candidates []SearchCandidate,
	topK int,
	similarityFunc func(a, b []float32) float64,
) []SearchResult {

	// Use object pool
	var scored []scoredResult
	if dr.useObjectPool {
		scored = make([]scoredResult, 0, len(candidates))
	}

	for _, candidate := range candidates {
		sim := similarityFunc(queryFull, candidate.Embedding)

		if dr.useObjectPool {
			sr := GetScoredResult()
			sr.result = SearchResult{
				Candidate:       candidate,
				Similarity:      sim,
				UsedReducedDims: false,
			}
			sr.similarity = sim
			scored = append(scored, *sr)
		} else {
			scored = append(scored, scoredResult{
				result: SearchResult{
					Candidate:       candidate,
					Similarity:      sim,
					UsedReducedDims: false,
				},
				similarity: sim,
			})
		}
	}

	// Use heap-based selection
	selector := NewTopKSelector(topK)
	results := selector.SelectTopKResults(scored)

	// Clean up pooled objects
	if dr.useObjectPool {
		for i := range scored {
			PutScoredResult(&scored[i])
		}
	}

	return results
}

// BatchHybridSearch performs multiple searches efficiently
func (dr *OptimizedDimensionReducer) BatchHybridSearch(
	ctx context.Context,
	queries [][]float32,
	candidateSets [][]SearchCandidate,
	topK int,
	similarityFunc func(a, b []float32) float64,
) ([][]SearchResult, error) {

	if len(queries) != len(candidateSets) {
		return nil, fmt.Errorf("queries and candidate sets must have same length")
	}

	// Process in parallel batches
	batchProcessor := NewBatchTopK(topK, 4) // 4 workers

	// Prepare queries
	topKQueries := make([]TopKQuery, len(queries))

	// Process each query
	var wg sync.WaitGroup
	errors := make([]error, len(queries))

	for i := range queries {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			// Get candidates scored by similarity
			scored := make([]scoredResult, 0, len(candidateSets[idx]))

			for _, candidate := range candidateSets[idx] {
				sim := similarityFunc(queries[idx], candidate.Embedding)
				scored = append(scored, scoredResult{
					result: SearchResult{
						Candidate:       candidate,
						Similarity:      sim,
						UsedReducedDims: false,
					},
					similarity: sim,
				})
			}

			topKQueries[idx] = TopKQuery{Candidates: scored}
		}(i)
	}

	wg.Wait()

	// Check for errors
	for _, err := range errors {
		if err != nil {
			return nil, err
		}
	}

	// Process batch
	results := batchProcessor.ProcessBatch(topKQueries)

	return results, nil
}

// GetMetrics returns current quality metrics (same as original)
func (dr *OptimizedDimensionReducer) GetMetrics() MetricsSnapshot {
	dr.metrics.mu.RLock()
	defer dr.metrics.mu.RUnlock()

	return MetricsSnapshot{
		TotalQueries:           dr.metrics.totalQueries,
		ReducedDimQueries:      dr.metrics.reducedDimQueries,
		FullDimReranks:         dr.metrics.fullDimReranks,
		AvgReductionTimeMs:     getFloat64FromAtomic(&dr.metrics.avgReductionTimeMs),
		AvgRerankTimeMs:        getFloat64FromAtomic(&dr.metrics.avgRerankTimeMs),
		HitRateBeforeReduction: getFloat64FromAtomic(&dr.metrics.hitRateBeforeReduction),
		HitRateAfterReduction:  getFloat64FromAtomic(&dr.metrics.hitRateAfterReduction),
		AccuracyScore:          getFloat64FromAtomic(&dr.metrics.accuracyScore),
		VarianceExplained:      getFloat64FromAtomic(&dr.metrics.varianceExplained),
		MemorySavedMB:          getFloat64FromAtomic(&dr.metrics.memorySavedMB),
	}
}

// GetReductionInfo returns dimension reduction information
func (dr *OptimizedDimensionReducer) GetReductionInfo() ReductionInfo {
	dr.mu.RLock()
	defer dr.mu.RUnlock()

	return ReductionInfo{
		OriginalDim:       dr.originalDim,
		ReducedDim:        dr.reducedDim,
		VarianceExplained: dr.calculateTotalVariance(),
		IsLearned:         dr.isLearned,
		CompressionRatio:  float64(dr.reducedDim) / float64(dr.originalDim),
	}
}

// Helper methods (reuse from original)
func (dr *OptimizedDimensionReducer) updateLearnMetrics(embeddings [][]float32, _ time.Duration) {
	dr.metrics.mu.Lock()
	defer dr.metrics.mu.Unlock()

	totalVariance := dr.calculateTotalVariance()
	setFloat64Atomic(&dr.metrics.varianceExplained, totalVariance)

	// Calculate memory saved
	originalSize := len(embeddings) * dr.originalDim * 4 // float32 = 4 bytes
	reducedSize := len(embeddings) * dr.reducedDim * 4
	saved := float64(originalSize-reducedSize) / (1024 * 1024)
	setFloat64Atomic(&dr.metrics.memorySavedMB, saved)
}

func (dr *OptimizedDimensionReducer) updateReductionMetrics(duration time.Duration) {
	dr.metrics.mu.Lock()
	defer dr.metrics.mu.Unlock()

	dr.metrics.reducedDimQueries++

	// Update rolling average
	n := float64(dr.metrics.reducedDimQueries)
	current := getFloat64FromAtomic(&dr.metrics.avgReductionTimeMs)
	newAvg := (current*(n-1) + float64(duration.Milliseconds())) / n
	setFloat64Atomic(&dr.metrics.avgReductionTimeMs, newAvg)
}

func (dr *OptimizedDimensionReducer) updateSearchMetrics(_ time.Duration, phase2Time time.Duration, _, finalCount int) {
	dr.metrics.mu.Lock()
	defer dr.metrics.mu.Unlock()

	dr.metrics.totalQueries++
	dr.metrics.fullDimReranks += int64(finalCount)

	// Update rolling average for rerank time
	n := float64(dr.metrics.fullDimReranks)
	if n > 0 {
		current := getFloat64FromAtomic(&dr.metrics.avgRerankTimeMs)
		newAvg := (current*(n-1) + float64(phase2Time.Milliseconds())) / n
		setFloat64Atomic(&dr.metrics.avgRerankTimeMs, newAvg)
	}
}

func (dr *OptimizedDimensionReducer) calculateTotalVariance() float64 {
	if len(dr.varianceRatio) == 0 {
		return 0
	}

	total := 0.0
	for _, v := range dr.varianceRatio {
		total += v
	}
	return total
}

// ReduceBatch reduces multiple embeddings efficiently
func (dr *OptimizedDimensionReducer) ReduceBatch(ctx context.Context, embeddings [][]float32) ([][]float32, error) {
	dr.mu.RLock()
	if !dr.isLearned {
		dr.mu.RUnlock()
		return nil, fmt.Errorf("reducer not learned yet")
	}
	dr.mu.RUnlock()

	return dr.reducer.Transform(ctx, embeddings)
}

// Helper functions for atomic float64 operations
func getFloat64FromAtomic(addr *uint64) float64 {
	return math.Float64frombits(atomic.LoadUint64(addr))
}

func setFloat64Atomic(addr *uint64, value float64) {
	atomic.StoreUint64(addr, math.Float64bits(value))
}
