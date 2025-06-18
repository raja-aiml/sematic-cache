package reduction

import (
	"context"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// DimensionReducer manages dimension reduction with quality monitoring
type DimensionReducer struct {
	mu            sync.RWMutex
	pca           *PCAReducer
	config        *Config
	metrics       *QualityMetrics
	originalDim   int
	reducedDim    int
	varianceRatio []float64
	isLearned     bool
}

// QualityMetrics tracks reduction quality and performance
type QualityMetrics struct {
	mu                     sync.RWMutex
	totalQueries           int64
	reducedDimQueries      int64
	fullDimReranks         int64
	avgReductionTimeMs     uint64 // Stored as uint64 for atomic operations
	avgRerankTimeMs        uint64 // Stored as uint64 for atomic operations
	hitRateBeforeReduction uint64 // Stored as uint64 for atomic operations
	hitRateAfterReduction  uint64 // Stored as uint64 for atomic operations
	accuracyScore          uint64 // Stored as uint64 for atomic operations
	varianceExplained      uint64 // Stored as uint64 for atomic operations
	memorySavedMB          uint64 // Stored as uint64 for atomic operations
}

// MetricsSnapshot is a thread-safe snapshot of quality metrics
type MetricsSnapshot struct {
	TotalQueries           int64
	ReducedDimQueries      int64
	FullDimReranks         int64
	AvgReductionTimeMs     float64
	AvgRerankTimeMs        float64
	HitRateBeforeReduction float64
	HitRateAfterReduction  float64
	AccuracyScore          float64
	VarianceExplained      float64
	MemorySavedMB          float64
}

// NewDimensionReducer creates a new dimension reducer with monitoring
func NewDimensionReducer(config *Config) (*DimensionReducer, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	return &DimensionReducer{
		config:  config,
		metrics: &QualityMetrics{},
		pca:     NewPCAReducer(config),
	}, nil
}

// Learn trains the reducer on sample embeddings with quality tracking
func (dr *DimensionReducer) Learn(ctx context.Context, embeddings [][]float32) error {
	if dr == nil {
		return fmt.Errorf("DimensionReducer is nil")
	}

	dr.mu.Lock()
	defer dr.mu.Unlock()

	// Validate embeddings
	if err := dr.validateEmbeddings(embeddings); err != nil {
		return fmt.Errorf("Learn: %w", err)
	}

	startTime := time.Now()

	// Train PCA
	if err := dr.pca.Fit(ctx, embeddings); err != nil {
		return fmt.Errorf("failed to fit PCA: %w", err)
	}

	dr.originalDim = dr.pca.OriginalDim()
	dr.reducedDim = dr.pca.ReducedDim()
	dr.varianceRatio = dr.pca.ExplainedVarianceRatio()
	dr.isLearned = true

	// Calculate quality metrics
	dr.updateLearnMetrics(embeddings, time.Since(startTime))

	return nil
}

// ReduceForSearch reduces embeddings for fast initial search
func (dr *DimensionReducer) ReduceForSearch(ctx context.Context, embedding []float32) ([]float32, error) {
	if dr == nil {
		return nil, fmt.Errorf("DimensionReducer is nil")
	}

	if embedding == nil {
		return nil, fmt.Errorf("embedding cannot be nil")
	}

	if len(embedding) == 0 {
		return nil, fmt.Errorf("embedding cannot be empty")
	}

	dr.mu.RLock()
	if !dr.isLearned {
		dr.mu.RUnlock()
		return nil, fmt.Errorf("reducer not learned yet")
	}

	// Validate embedding dimension
	if len(embedding) != dr.originalDim {
		dr.mu.RUnlock()
		return nil, fmt.Errorf("embedding dimension mismatch: expected %d, got %d", dr.originalDim, len(embedding))
	}
	dr.mu.RUnlock()

	startTime := time.Now()
	reduced, err := dr.pca.Transform(ctx, [][]float32{embedding})
	if err != nil {
		return nil, err
	}

	dr.updateReductionMetrics(time.Since(startTime))
	return reduced[0], nil
}

// HybridSearch performs fast search with reduced dims then re-ranks with full dims
func (dr *DimensionReducer) HybridSearch(
	ctx context.Context,
	queryEmbedding []float32,
	candidates []SearchCandidate,
	topK int,
	similarityFunc func(a, b []float32) float64,
) ([]SearchResult, error) {
	if dr == nil {
		return nil, fmt.Errorf("DimensionReducer is nil")
	}

	// Validate inputs
	if err := dr.validateSearchInputs(queryEmbedding, candidates, topK, similarityFunc); err != nil {
		return nil, fmt.Errorf("HybridSearch: %w", err)
	}

	if !dr.isLearned {
		// Fallback to full dimension search
		return dr.fullDimensionSearch(queryEmbedding, candidates, topK, similarityFunc), nil
	}

	// Phase 1: Fast search with reduced dimensions
	reducedQuery, err := dr.ReduceForSearch(ctx, queryEmbedding)
	if err != nil {
		return nil, fmt.Errorf("failed to reduce query: %w", err)
	}

	startPhase1 := time.Now()

	// Get 3x topK candidates using reduced dimensions for better coverage
	phase1Candidates := dr.searchReduced(reducedQuery, candidates, topK*3, similarityFunc)

	phase1Time := time.Since(startPhase1)

	// Phase 2: Re-rank top candidates with full dimensions
	startPhase2 := time.Now()
	finalResults := dr.rerankWithFullDims(queryEmbedding, phase1Candidates, topK, similarityFunc)
	phase2Time := time.Since(startPhase2)

	// Update metrics
	dr.updateSearchMetrics(phase1Time, phase2Time, len(phase1Candidates), topK)

	return finalResults, nil
}

// SearchCandidate represents a candidate for similarity search
type SearchCandidate struct {
	ID               string
	Embedding        []float32
	ReducedEmbedding []float32
	Metadata         map[string]interface{}
}

// SearchResult represents a search result with similarity score
type SearchResult struct {
	Candidate       SearchCandidate
	Similarity      float64
	UsedReducedDims bool
}

// scoredCandidate holds a candidate with its similarity score
type scoredCandidate struct {
	candidate  SearchCandidate
	similarity float64
}

// scoredResult holds a result with its similarity score
type scoredResult struct {
	result     SearchResult
	similarity float64
}

// searchReduced performs similarity search using reduced dimensions
func (dr *DimensionReducer) searchReduced(
	queryReduced []float32,
	candidates []SearchCandidate,
	topK int,
	similarityFunc func(a, b []float32) float64,
) []SearchCandidate {
	if queryReduced == nil || len(queryReduced) == 0 {
		return []SearchCandidate{}
	}

	scored := make([]scoredCandidate, 0, len(candidates))

	for _, candidate := range candidates {
		if len(candidate.ReducedEmbedding) == 0 {
			continue
		}

		// Validate reduced embedding dimension
		if len(candidate.ReducedEmbedding) != len(queryReduced) {
			continue // Skip mismatched dimensions
		}

		sim := similarityFunc(queryReduced, candidate.ReducedEmbedding)
		scored = append(scored, scoredCandidate{
			candidate:  candidate,
			similarity: sim,
		})
	}

	// Use heap-based selection for better performance
	if len(scored) > topK*2 { // Only use heap for larger datasets
		selector := NewTopKSelector(topK)
		return selector.SelectTopK(scored)
	}

	// For small datasets, sorting is fine
	sortBySimilarity(scored)

	// Return top candidates
	if topK > len(scored) {
		topK = len(scored)
	}

	results := make([]SearchCandidate, topK)
	for i := 0; i < topK; i++ {
		results[i] = scored[i].candidate
	}

	return results
}

// rerankWithFullDims re-ranks candidates using full dimension embeddings
func (dr *DimensionReducer) rerankWithFullDims(
	queryFull []float32,
	candidates []SearchCandidate,
	topK int,
	similarityFunc func(a, b []float32) float64,
) []SearchResult {
	if queryFull == nil || len(queryFull) == 0 {
		return []SearchResult{}
	}

	scored := make([]scoredResult, 0, len(candidates))

	for _, candidate := range candidates {
		// Validate embedding dimension
		if len(candidate.Embedding) != len(queryFull) {
			continue // Skip mismatched dimensions
		}

		sim := similarityFunc(queryFull, candidate.Embedding)
		scored = append(scored, scoredResult{
			result: SearchResult{
				Candidate:       candidate,
				Similarity:      sim,
				UsedReducedDims: false,
			},
			similarity: sim,
		})
	}

	// Use heap-based selection for better performance
	if len(scored) > topK*2 { // Only use heap for larger datasets
		selector := NewTopKSelector(topK)
		return selector.SelectTopKResults(scored)
	}

	// For small datasets, sorting is fine
	sortResultsBySimilarity(scored)

	// Return top K
	if topK > len(scored) {
		topK = len(scored)
	}

	results := make([]SearchResult, topK)
	for i := 0; i < topK; i++ {
		results[i] = scored[i].result
	}

	return results
}

// fullDimensionSearch performs search using only full dimensions (fallback)
func (dr *DimensionReducer) fullDimensionSearch(
	queryFull []float32,
	candidates []SearchCandidate,
	topK int,
	similarityFunc func(a, b []float32) float64,
) []SearchResult {
	if queryFull == nil || len(queryFull) == 0 {
		return []SearchResult{}
	}

	scored := make([]scoredResult, 0, len(candidates))

	for _, candidate := range candidates {
		// Validate embedding dimension
		if len(candidate.Embedding) != len(queryFull) {
			continue // Skip mismatched dimensions
		}

		sim := similarityFunc(queryFull, candidate.Embedding)
		scored = append(scored, scoredResult{
			result: SearchResult{
				Candidate:       candidate,
				Similarity:      sim,
				UsedReducedDims: false,
			},
			similarity: sim,
		})
	}

	// Sort by similarity descending
	sortResultsBySimilarity(scored)

	// Return top K
	if topK > len(scored) {
		topK = len(scored)
	}

	results := make([]SearchResult, topK)
	for i := 0; i < topK; i++ {
		results[i] = scored[i].result
	}

	return results
}

// GetMetrics returns current quality metrics
func (dr *DimensionReducer) GetMetrics() MetricsSnapshot {
	if dr == nil || dr.metrics == nil {
		return MetricsSnapshot{}
	}
	// Read all atomic values
	return MetricsSnapshot{
		TotalQueries:           atomic.LoadInt64(&dr.metrics.totalQueries),
		ReducedDimQueries:      atomic.LoadInt64(&dr.metrics.reducedDimQueries),
		FullDimReranks:         atomic.LoadInt64(&dr.metrics.fullDimReranks),
		AvgReductionTimeMs:     math.Float64frombits(atomic.LoadUint64(&dr.metrics.avgReductionTimeMs)),
		AvgRerankTimeMs:        math.Float64frombits(atomic.LoadUint64(&dr.metrics.avgRerankTimeMs)),
		HitRateBeforeReduction: math.Float64frombits(atomic.LoadUint64(&dr.metrics.hitRateBeforeReduction)),
		HitRateAfterReduction:  math.Float64frombits(atomic.LoadUint64(&dr.metrics.hitRateAfterReduction)),
		AccuracyScore:          math.Float64frombits(atomic.LoadUint64(&dr.metrics.accuracyScore)),
		VarianceExplained:      math.Float64frombits(atomic.LoadUint64(&dr.metrics.varianceExplained)),
		MemorySavedMB:          math.Float64frombits(atomic.LoadUint64(&dr.metrics.memorySavedMB)),
	}
}

// GetReductionInfo returns dimension reduction information
func (dr *DimensionReducer) GetReductionInfo() ReductionInfo {
	if dr == nil {
		return ReductionInfo{}
	}

	dr.mu.RLock()
	defer dr.mu.RUnlock()

	compressionRatio := 0.0
	if dr.originalDim > 0 {
		compressionRatio = float64(dr.reducedDim) / float64(dr.originalDim)
	}

	return ReductionInfo{
		OriginalDim:       dr.originalDim,
		ReducedDim:        dr.reducedDim,
		VarianceExplained: dr.calculateTotalVariance(),
		IsLearned:         dr.isLearned,
		CompressionRatio:  compressionRatio,
	}
}

// ReductionInfo contains dimension reduction information
type ReductionInfo struct {
	OriginalDim       int
	ReducedDim        int
	VarianceExplained float64
	IsLearned         bool
	CompressionRatio  float64
}

// Helper functions for metrics updates
func (dr *DimensionReducer) updateLearnMetrics(embeddings [][]float32, _ time.Duration) {
	totalVariance := dr.calculateTotalVariance()
	atomic.StoreUint64(&dr.metrics.varianceExplained, math.Float64bits(totalVariance))

	// Calculate memory saved
	originalSize := len(embeddings) * dr.originalDim * 4 // float32 = 4 bytes
	reducedSize := len(embeddings) * dr.reducedDim * 4
	memorySaved := float64(originalSize-reducedSize) / (1024 * 1024)
	atomic.StoreUint64(&dr.metrics.memorySavedMB, math.Float64bits(memorySaved))
}

func (dr *DimensionReducer) updateReductionMetrics(duration time.Duration) {
	// Increment queries atomically
	newCount := atomic.AddInt64(&dr.metrics.reducedDimQueries, 1)

	// Update rolling average atomically
	for {
		oldAvg := atomic.LoadUint64(&dr.metrics.avgReductionTimeMs)
		oldAvgFloat := math.Float64frombits(oldAvg)
		newAvgFloat := (oldAvgFloat*float64(newCount-1) + float64(duration.Milliseconds())) / float64(newCount)
		newAvg := math.Float64bits(newAvgFloat)
		if atomic.CompareAndSwapUint64(&dr.metrics.avgReductionTimeMs, oldAvg, newAvg) {
			break
		}
	}
}

func (dr *DimensionReducer) updateSearchMetrics(_ time.Duration, phase2Time time.Duration, _, finalCount int) {
	// Increment counters atomically
	atomic.AddInt64(&dr.metrics.totalQueries, 1)
	newReranks := atomic.AddInt64(&dr.metrics.fullDimReranks, int64(finalCount))

	// Update rolling average for rerank time atomically
	if newReranks > 0 {
		for {
			oldAvg := atomic.LoadUint64(&dr.metrics.avgRerankTimeMs)
			oldAvgFloat := math.Float64frombits(oldAvg)
			newAvgFloat := (oldAvgFloat*float64(newReranks-int64(finalCount)) + float64(phase2Time.Milliseconds())) / float64(newReranks)
			newAvg := math.Float64bits(newAvgFloat)
			if atomic.CompareAndSwapUint64(&dr.metrics.avgRerankTimeMs, oldAvg, newAvg) {
				break
			}
		}
	}
}

func (dr *DimensionReducer) calculateTotalVariance() float64 {
	if len(dr.varianceRatio) == 0 {
		return 0
	}

	total := 0.0
	for _, v := range dr.varianceRatio {
		total += v
	}
	return total
}

// UpdateHitRates updates cache hit rate metrics for A/B testing
func (dr *DimensionReducer) UpdateHitRates(beforeReduction, afterReduction float64) {
	if dr == nil || dr.metrics == nil {
		return
	}

	// Validate input rates
	if beforeReduction < 0 || beforeReduction > 1 {
		return // Invalid hit rate
	}
	if afterReduction < 0 || afterReduction > 1 {
		return // Invalid hit rate
	}

	// Update hit rates atomically
	atomic.StoreUint64(&dr.metrics.hitRateBeforeReduction, math.Float64bits(beforeReduction))
	atomic.StoreUint64(&dr.metrics.hitRateAfterReduction, math.Float64bits(afterReduction))
	
	// Calculate and store accuracy score
	accuracy := 0.0
	if beforeReduction > 0 {
		accuracy = afterReduction / beforeReduction
	}
	atomic.StoreUint64(&dr.metrics.accuracyScore, math.Float64bits(accuracy))
}

// Helper sorting functions
func sortBySimilarity(candidates []scoredCandidate) {
	// Simple insertion sort for now - can be optimized with heap for large datasets
	for i := 1; i < len(candidates); i++ {
		j := i
		for j > 0 && candidates[j].similarity > candidates[j-1].similarity {
			candidates[j], candidates[j-1] = candidates[j-1], candidates[j]
			j--
		}
	}
}

func sortResultsBySimilarity(results []scoredResult) {
	for i := 1; i < len(results); i++ {
		j := i
		for j > 0 && results[j].similarity > results[j-1].similarity {
			results[j], results[j-1] = results[j-1], results[j]
			j--
		}
	}
}

// ReduceBatch reduces multiple embeddings efficiently
func (dr *DimensionReducer) ReduceBatch(ctx context.Context, embeddings [][]float32) ([][]float32, error) {
	if dr == nil {
		return nil, fmt.Errorf("DimensionReducer is nil")
	}

	if err := dr.validateEmbeddings(embeddings); err != nil {
		return nil, fmt.Errorf("ReduceBatch: %w", err)
	}

	dr.mu.RLock()
	if !dr.isLearned {
		dr.mu.RUnlock()
		return nil, fmt.Errorf("reducer not learned yet")
	}
	dr.mu.RUnlock()

	return dr.pca.Transform(ctx, embeddings)
}

// ShouldUseReduction determines if reduction should be used based on metrics
func (dr *DimensionReducer) ShouldUseReduction() bool {
	if dr == nil || dr.metrics == nil {
		return false
	}

	info := dr.GetReductionInfo()
	accuracyScore := math.Float64frombits(atomic.LoadUint64(&dr.metrics.accuracyScore))

	// Use reduction if:
	// 1. Reducer is learned
	// 2. Variance explained is above threshold (95%)
	// 3. Accuracy score is above threshold (90%)
	// 4. Significant compression ratio (< 0.5)
	return info.IsLearned &&
		info.VarianceExplained >= 0.95 &&
		(accuracyScore >= 0.9 || accuracyScore == 0) &&
		info.CompressionRatio <= 0.5
}

// EstimateSearchSpeedup estimates the speedup from dimension reduction
func (dr *DimensionReducer) EstimateSearchSpeedup() float64 {
	if dr == nil {
		return 1.0
	}

	info := dr.GetReductionInfo()
	if !info.IsLearned || info.CompressionRatio == 0 {
		return 1.0
	}

	// Search time is roughly proportional to dimensions
	// Account for re-ranking overhead (assume 30% of candidates re-ranked)
	baseSpeedup := 1.0 / info.CompressionRatio
	rerankOverhead := 0.3

	return baseSpeedup*(1-rerankOverhead) + rerankOverhead
}

// validateEmbeddings validates embeddings slice
func (dr *DimensionReducer) validateEmbeddings(embeddings [][]float32) error {
	if embeddings == nil {
		return fmt.Errorf("embeddings cannot be nil")
	}

	if len(embeddings) == 0 {
		return fmt.Errorf("no embeddings provided")
	}

	// Check first embedding
	if embeddings[0] == nil || len(embeddings[0]) == 0 {
		return fmt.Errorf("first embedding is empty")
	}

	dim := len(embeddings[0])

	// Validate all embeddings have same dimension
	for i, emb := range embeddings {
		if emb == nil {
			return fmt.Errorf("embedding %d is nil", i)
		}
		if len(emb) != dim {
			return fmt.Errorf("inconsistent dimensions: embedding %d has %d dimensions, expected %d", i, len(emb), dim)
		}

		// Check for invalid values
		for j, val := range emb {
			if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
				return fmt.Errorf("embedding %d contains invalid value at index %d: %v", i, j, val)
			}
		}
	}

	return nil
}

// validateSearchInputs validates inputs for search operations
func (dr *DimensionReducer) validateSearchInputs(
	queryEmbedding []float32,
	candidates []SearchCandidate,
	topK int,
	similarityFunc func(a, b []float32) float64,
) error {
	if queryEmbedding == nil {
		return fmt.Errorf("query embedding cannot be nil")
	}

	if len(queryEmbedding) == 0 {
		return fmt.Errorf("query embedding cannot be empty")
	}

	// Check for invalid values in query
	for i, val := range queryEmbedding {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			return fmt.Errorf("query embedding contains invalid value at index %d: %v", i, val)
		}
	}

	if candidates == nil {
		return fmt.Errorf("candidates cannot be nil")
	}

	if topK <= 0 {
		return fmt.Errorf("topK must be positive, got %d", topK)
	}

	if similarityFunc == nil {
		return fmt.Errorf("similarity function cannot be nil")
	}

	return nil
}
