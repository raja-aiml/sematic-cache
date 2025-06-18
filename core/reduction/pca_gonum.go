package reduction

import (
	"context"
	"fmt"
	"sync"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// PCAGonumReducer implements Principal Component Analysis using Gonum for optimal performance
type PCAGonumReducer struct {
	mu                sync.RWMutex
	config            *Config
	mean              []float64
	components        *mat.Dense
	explainedVariance []float64
	singularValues    []float64
	originalDim       int
	reducedDim        int
	isFitted          bool
}

// NewPCAGonumReducer creates a new PCA reducer with Gonum optimization
func NewPCAGonumReducer(config *Config) *PCAGonumReducer {
	return &PCAGonumReducer{
		config: config,
	}
}

// Fit learns the PCA transformation from training data using optimized SVD
func (p *PCAGonumReducer) Fit(ctx context.Context, embeddings [][]float32) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(embeddings) == 0 {
		return fmt.Errorf("no embeddings provided")
	}

	// Check context
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	n := len(embeddings)
	d := len(embeddings[0])
	p.originalDim = d

	// Validate dimensions
	for i, emb := range embeddings {
		if len(emb) != d {
			return fmt.Errorf("inconsistent dimensions: embedding %d has %d dims, expected %d", i, len(emb), d)
		}
	}

	// Convert float32 to float64 matrix
	data := mat.NewDense(n, d, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			data.Set(i, j, float64(embeddings[i][j]))
		}
	}

	// Step 1: Calculate mean and center the data
	p.mean = make([]float64, d)
	for j := 0; j < d; j++ {
		col := mat.Col(nil, j, data)
		p.mean[j] = stat.Mean(col, nil)
	}

	// Center the data
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			data.Set(i, j, data.At(i, j)-p.mean[j])
		}
	}

	// Step 2: Perform SVD using Gonum's optimized implementation
	var svd mat.SVD
	ok := svd.Factorize(data, mat.SVDThin)
	if !ok {
		return fmt.Errorf("SVD factorization failed")
	}

	// Get singular values
	singularValues := svd.Values(nil)
	p.singularValues = singularValues

	// Calculate explained variance
	totalVariance := 0.0
	variances := make([]float64, len(singularValues))
	for i, s := range singularValues {
		variances[i] = (s * s) / float64(n-1)
		totalVariance += variances[i]
	}

	// Calculate explained variance ratio
	p.explainedVariance = make([]float64, len(variances))
	for i, v := range variances {
		p.explainedVariance[i] = v / totalVariance
	}

	// Determine number of components to keep
	p.reducedDim = p.determineNumComponents()
	if p.reducedDim > len(singularValues) {
		p.reducedDim = len(singularValues)
	}

	// Get the right singular vectors (components)
	vt := new(mat.Dense)
	svd.VTo(vt)

	// Keep only the top components (transpose to get components as rows)
	p.components = mat.NewDense(p.reducedDim, d, nil)
	for i := 0; i < p.reducedDim; i++ {
		for j := 0; j < d; j++ {
			p.components.Set(i, j, vt.At(j, i))
		}
	}

	p.isFitted = true
	return nil
}

// Transform reduces dimensionality of embeddings using the fitted PCA model
func (p *PCAGonumReducer) Transform(ctx context.Context, embeddings [][]float32) ([][]float32, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if !p.isFitted {
		return nil, fmt.Errorf("PCA not fitted yet")
	}

	if len(embeddings) == 0 {
		return [][]float32{}, nil
	}

	// Validate dimensions
	for i, emb := range embeddings {
		if len(emb) != p.originalDim {
			return nil, fmt.Errorf("embedding %d has %d dimensions, expected %d", i, len(emb), p.originalDim)
		}
	}

	result := make([][]float32, len(embeddings))

	for i, emb := range embeddings {
		// Check context
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Center the embedding
		centered := mat.NewVecDense(len(emb), nil)
		for j, val := range emb {
			centered.SetVec(j, float64(val)-p.mean[j])
		}

		// Project onto principal components
		reduced := mat.NewVecDense(p.reducedDim, nil)
		reduced.MulVec(p.components, centered)

		// Convert back to float32
		result[i] = make([]float32, p.reducedDim)
		for j := 0; j < p.reducedDim; j++ {
			result[i][j] = float32(reduced.AtVec(j))
		}
	}

	return result, nil
}

// FitTransform combines fit and transform in one optimized operation
func (p *PCAGonumReducer) FitTransform(ctx context.Context, embeddings [][]float32) ([][]float32, error) {
	if err := p.Fit(ctx, embeddings); err != nil {
		return nil, err
	}
	return p.Transform(ctx, embeddings)
}

// InverseTransform reconstructs embeddings from reduced dimensions
func (p *PCAGonumReducer) InverseTransform(ctx context.Context, reduced [][]float32) ([][]float32, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if !p.isFitted {
		return nil, fmt.Errorf("PCA not fitted yet")
	}

	result := make([][]float32, len(reduced))

	for i, red := range reduced {
		if len(red) != p.reducedDim {
			return nil, fmt.Errorf("reduced embedding %d has %d dimensions, expected %d", i, len(red), p.reducedDim)
		}

		// Convert to vector
		redVec := mat.NewVecDense(len(red), nil)
		for j, val := range red {
			redVec.SetVec(j, float64(val))
		}

		// Reconstruct: multiply by components transpose and add mean
		reconstructed := mat.NewVecDense(p.originalDim, nil)
		componentsT := p.components.T()
		reconstructed.MulVec(componentsT, redVec)

		// Add back the mean and convert to float32
		result[i] = make([]float32, p.originalDim)
		for j := 0; j < p.originalDim; j++ {
			result[i][j] = float32(reconstructed.AtVec(j) + p.mean[j])
		}
	}

	return result, nil
}

// determineNumComponents determines optimal number of components based on config
func (p *PCAGonumReducer) determineNumComponents() int {
	if p.config.VarianceThreshold > 0 {
		// Find number of components for desired variance
		cumSum := 0.0
		for i, v := range p.explainedVariance {
			cumSum += v
			if cumSum >= p.config.VarianceThreshold {
				return i + 1
			}
		}
		return len(p.explainedVariance)
	}

	// Use target dimension
	if p.config.TargetDim < len(p.explainedVariance) {
		return p.config.TargetDim
	}
	return len(p.explainedVariance)
}

// OriginalDim returns the original embedding dimension
func (p *PCAGonumReducer) OriginalDim() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.originalDim
}

// ReducedDim returns the reduced embedding dimension
func (p *PCAGonumReducer) ReducedDim() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.reducedDim
}

// ExplainedVarianceRatio returns variance explained by each component
func (p *PCAGonumReducer) ExplainedVarianceRatio() []float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.explainedVariance) > p.reducedDim {
		return p.explainedVariance[:p.reducedDim]
	}
	return p.explainedVariance
}

// GetReconstructionError calculates the reconstruction error for quality monitoring
func (p *PCAGonumReducer) GetReconstructionError(ctx context.Context, original [][]float32) (float64, error) {
	// Transform and inverse transform
	reduced, err := p.Transform(ctx, original)
	if err != nil {
		return 0, err
	}

	reconstructed, err := p.InverseTransform(ctx, reduced)
	if err != nil {
		return 0, err
	}

	// Calculate mean squared error
	totalError := 0.0
	totalElements := 0
	for i := range original {
		for j := range original[i] {
			diff := float64(original[i][j] - reconstructed[i][j])
			totalError += diff * diff
			totalElements++
		}
	}

	return totalError / float64(totalElements), nil
}

// BatchTransform efficiently transforms multiple embeddings in parallel
func (p *PCAGonumReducer) BatchTransform(ctx context.Context, embeddings [][]float32, batchSize int) ([][]float32, error) {
	if batchSize <= 0 {
		batchSize = 100
	}

	p.mu.RLock()
	defer p.mu.RUnlock()

	if !p.isFitted {
		return nil, fmt.Errorf("PCA not fitted yet")
	}

	n := len(embeddings)
	if n == 0 {
		return [][]float32{}, nil
	}

	// For small datasets, don't use batching
	if n <= batchSize {
		return p.Transform(ctx, embeddings)
	}

	// Convert all embeddings to a matrix for efficient computation
	data := mat.NewDense(n, p.originalDim, nil)
	for i := 0; i < n; i++ {
		if len(embeddings[i]) != p.originalDim {
			return nil, fmt.Errorf("embedding %d has incorrect dimensions", i)
		}
		for j := 0; j < p.originalDim; j++ {
			data.Set(i, j, float64(embeddings[i][j])-p.mean[j])
		}
	}

	// Perform matrix multiplication: data * components^T
	result := mat.NewDense(n, p.reducedDim, nil)
	result.Mul(data, p.components.T())

	// Convert back to float32
	output := make([][]float32, n)
	for i := 0; i < n; i++ {
		output[i] = make([]float32, p.reducedDim)
		for j := 0; j < p.reducedDim; j++ {
			output[i][j] = float32(result.At(i, j))
		}
	}

	return output, nil
}

// EstimateMemorySavings calculates the memory savings from dimension reduction
func (p *PCAGonumReducer) EstimateMemorySavings(numEmbeddings int) (originalMB, reducedMB, savingsMB float64) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if !p.isFitted {
		return 0, 0, 0
	}

	// float32 = 4 bytes
	originalMB = float64(numEmbeddings*p.originalDim*4) / (1024 * 1024)
	reducedMB = float64(numEmbeddings*p.reducedDim*4) / (1024 * 1024)
	savingsMB = originalMB - reducedMB

	return originalMB, reducedMB, savingsMB
}