// Package reduction provides dimension reduction algorithms for embeddings.
// This significantly improves search performance while maintaining semantic accuracy.
package reduction

import (
	"context"
	"fmt"
)

// Reducer defines the interface for dimension reduction algorithms.
// Implementations should be thread-safe for concurrent use.
type Reducer interface {
	// Fit trains the reduction model on sample embeddings
	Fit(ctx context.Context, embeddings [][]float32) error

	// Transform reduces the dimensionality of embeddings
	Transform(ctx context.Context, embeddings [][]float32) ([][]float32, error)

	// FitTransform combines Fit and Transform in one operation
	FitTransform(ctx context.Context, embeddings [][]float32) ([][]float32, error)

	// InverseTransform reconstructs original dimensions (if supported)
	InverseTransform(ctx context.Context, reduced [][]float32) ([][]float32, error)

	// OriginalDim returns the original embedding dimension
	OriginalDim() int

	// ReducedDim returns the reduced embedding dimension
	ReducedDim() int

	// ExplainedVarianceRatio returns the variance explained by each component
	ExplainedVarianceRatio() []float64
}

// Config holds common configuration for reduction algorithms
type Config struct {
	// TargetDim is the desired output dimension
	TargetDim int

	// VarianceThreshold is the minimum variance to retain (0.0-1.0)
	// If set, overrides TargetDim
	VarianceThreshold float64

	// Standardize whether to standardize features before reduction
	Standardize bool

	// RandomSeed for reproducible results
	RandomSeed int64
}

// Validate checks if the configuration is valid
func (c *Config) Validate() error {
	if c.TargetDim <= 0 && c.VarianceThreshold <= 0 {
		return fmt.Errorf("either TargetDim or VarianceThreshold must be positive")
	}
	if c.VarianceThreshold < 0 || c.VarianceThreshold > 1 {
		return fmt.Errorf("VarianceThreshold must be between 0 and 1")
	}
	return nil
}

// ReductionMetrics contains performance metrics for dimension reduction
type ReductionMetrics struct {
	// OriginalDim is the original embedding dimension
	OriginalDim int

	// ReducedDim is the reduced embedding dimension
	ReducedDim int

	// CompressionRatio is ReducedDim/OriginalDim
	CompressionRatio float64

	// TotalVarianceExplained is the cumulative explained variance
	TotalVarianceExplained float64

	// ReductionTimeMs is the time taken for reduction in milliseconds
	ReductionTimeMs float64

	// MemorySavedBytes estimates memory saved by reduction
	MemorySavedBytes int64
}

// Calculate computes derived metrics
func (m *ReductionMetrics) Calculate() {
	if m.OriginalDim > 0 {
		m.CompressionRatio = float64(m.ReducedDim) / float64(m.OriginalDim)
		m.MemorySavedBytes = int64((m.OriginalDim - m.ReducedDim) * 4) // float32 = 4 bytes
	}
}
