package reduction

import (
	"context"
	"fmt"
	"sync"
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
	avgReductionTimeMs     float64
	avgRerankTimeMs        float64
	hitRateBeforeReduction float64
	hitRateAfterReduction  float64
	accuracyScore          float64
	varianceExplained      float64
	memorySavedMB          float64
}

// NewDimensionReducer creates a new dimension reducer with monitoring
func NewDimensionReducer(config *Config) (*DimensionReducer, error) {
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
	dr.mu.Lock()
	defer dr.mu.Unlock()

	if len(embeddings) == 0 {
		return fmt.Errorf("no embeddings provided")
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
	dr.mu.RLock()
	if !dr.isLearned {
		dr.mu.RUnlock()
		return nil, fmt.Errorf("reducer not learned yet")
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

	scored := make([]scoredCandidate, 0, len(candidates))

	for _, candidate := range candidates {
		if len(candidate.ReducedEmbedding) == 0 {
			continue
		}

		sim := similarityFunc(queryReduced, candidate.ReducedEmbedding)
		scored = append(scored, scoredCandidate{
			candidate:  candidate,
			similarity: sim,
		})
	}

	// Sort by similarity descending
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

	scored := make([]scoredResult, 0, len(candidates))

	for _, candidate := range candidates {
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

// fullDimensionSearch performs search using only full dimensions (fallback)
func (dr *DimensionReducer) fullDimensionSearch(
	queryFull []float32,
	candidates []SearchCandidate,
	topK int,
	similarityFunc func(a, b []float32) float64,
) []SearchResult {

	scored := make([]scoredResult, 0, len(candidates))

	for _, candidate := range candidates {
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
func (dr *DimensionReducer) GetMetrics() QualityMetrics {
	dr.metrics.mu.RLock()
	defer dr.metrics.mu.RUnlock()

	// Create a copy without the mutex
	return QualityMetrics{
		totalQueries:           dr.metrics.totalQueries,
		reducedDimQueries:      dr.metrics.reducedDimQueries,
		fullDimReranks:         dr.metrics.fullDimReranks,
		avgReductionTimeMs:     dr.metrics.avgReductionTimeMs,
		avgRerankTimeMs:        dr.metrics.avgRerankTimeMs,
		hitRateBeforeReduction: dr.metrics.hitRateBeforeReduction,
		hitRateAfterReduction:  dr.metrics.hitRateAfterReduction,
		accuracyScore:          dr.metrics.accuracyScore,
		varianceExplained:      dr.metrics.varianceExplained,
		memorySavedMB:          dr.metrics.memorySavedMB,
	}
}

// GetReductionInfo returns dimension reduction information
func (dr *DimensionReducer) GetReductionInfo() ReductionInfo {
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

// ReductionInfo contains dimension reduction information
type ReductionInfo struct {
	OriginalDim       int
	ReducedDim        int
	VarianceExplained float64
	IsLearned         bool
	CompressionRatio  float64
}

// Helper functions for metrics updates
func (dr *DimensionReducer) updateLearnMetrics(embeddings [][]float32, duration time.Duration) {
	dr.metrics.mu.Lock()
	defer dr.metrics.mu.Unlock()

	totalVariance := dr.calculateTotalVariance()
	dr.metrics.varianceExplained = totalVariance

	// Calculate memory saved
	originalSize := len(embeddings) * dr.originalDim * 4 // float32 = 4 bytes
	reducedSize := len(embeddings) * dr.reducedDim * 4
	dr.metrics.memorySavedMB = float64(originalSize-reducedSize) / (1024 * 1024)
}

func (dr *DimensionReducer) updateReductionMetrics(duration time.Duration) {
	dr.metrics.mu.Lock()
	defer dr.metrics.mu.Unlock()

	dr.metrics.reducedDimQueries++

	// Update rolling average
	n := float64(dr.metrics.reducedDimQueries)
	dr.metrics.avgReductionTimeMs = (dr.metrics.avgReductionTimeMs*(n-1) + float64(duration.Milliseconds())) / n
}

func (dr *DimensionReducer) updateSearchMetrics(phase1Time, phase2Time time.Duration, phase1Count, finalCount int) {
	dr.metrics.mu.Lock()
	defer dr.metrics.mu.Unlock()

	dr.metrics.totalQueries++
	dr.metrics.fullDimReranks += int64(finalCount)

	// Update rolling average for rerank time
	n := float64(dr.metrics.fullDimReranks)
	if n > 0 {
		dr.metrics.avgRerankTimeMs = (dr.metrics.avgRerankTimeMs*(n-1) + float64(phase2Time.Milliseconds())) / n
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
	dr.metrics.mu.Lock()
	defer dr.metrics.mu.Unlock()

	dr.metrics.hitRateBeforeReduction = beforeReduction
	dr.metrics.hitRateAfterReduction = afterReduction
	dr.metrics.accuracyScore = afterReduction / beforeReduction
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
	metrics := dr.GetMetrics()
	info := dr.GetReductionInfo()

	// Use reduction if:
	// 1. Reducer is learned
	// 2. Variance explained is above threshold (95%)
	// 3. Accuracy score is above threshold (90%)
	// 4. Significant compression ratio (< 0.5)
	return info.IsLearned &&
		info.VarianceExplained >= 0.95 &&
		(metrics.accuracyScore >= 0.9 || metrics.accuracyScore == 0) &&
		info.CompressionRatio <= 0.5
}

// EstimateSearchSpeedup estimates the speedup from dimension reduction
func (dr *DimensionReducer) EstimateSearchSpeedup() float64 {
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
