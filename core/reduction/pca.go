package reduction

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
)

// PCAReducer implements Principal Component Analysis for dimension reduction
type PCAReducer struct {
	mu                sync.RWMutex
	config            *Config
	mean              []float64
	components        [][]float64
	explainedVariance []float64
	originalDim       int
	reducedDim        int
	isFitted          bool
}

// NewPCAReducer creates a new PCA reducer
func NewPCAReducer(config *Config) *PCAReducer {
	return &PCAReducer{
		config: config,
	}
}

// Fit learns the PCA transformation from training data
func (p *PCAReducer) Fit(ctx context.Context, embeddings [][]float32) error {
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

	// Step 1: Calculate mean
	p.mean = make([]float64, d)
	for _, emb := range embeddings {
		for j, val := range emb {
			p.mean[j] += float64(val)
		}
	}
	for j := range p.mean {
		p.mean[j] /= float64(n)
	}

	// Step 2: Center the data
	centered := make([][]float64, n)
	for i, emb := range embeddings {
		centered[i] = make([]float64, d)
		for j, val := range emb {
			centered[i][j] = float64(val) - p.mean[j]
		}
	}

	// Step 3: Compute covariance matrix (using SVD approach for efficiency)
	// For large dimensions, we use truncated SVD
	components, singularValues, err := p.truncatedSVD(ctx, centered)
	if err != nil {
		return fmt.Errorf("SVD failed: %w", err)
	}

	// Step 4: Calculate explained variance
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

	// Step 5: Determine number of components to keep
	p.reducedDim = p.determineNumComponents()

	// Keep only the top components
	p.components = make([][]float64, p.reducedDim)
	for i := 0; i < p.reducedDim; i++ {
		p.components[i] = components[i]
	}

	p.isFitted = true
	return nil
}

// Transform reduces dimensionality of embeddings
func (p *PCAReducer) Transform(ctx context.Context, embeddings [][]float32) ([][]float32, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if !p.isFitted {
		return nil, fmt.Errorf("PCA not fitted yet")
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
		centered := make([]float64, len(emb))
		for j, val := range emb {
			centered[j] = float64(val) - p.mean[j]
		}

		// Project onto principal components
		reduced := make([]float32, p.reducedDim)
		for j := 0; j < p.reducedDim; j++ {
			dot := 0.0
			for k, val := range centered {
				dot += val * p.components[j][k]
			}
			reduced[j] = float32(dot)
		}

		result[i] = reduced
	}

	return result, nil
}

// FitTransform combines fit and transform
func (p *PCAReducer) FitTransform(ctx context.Context, embeddings [][]float32) ([][]float32, error) {
	if err := p.Fit(ctx, embeddings); err != nil {
		return nil, err
	}
	return p.Transform(ctx, embeddings)
}

// InverseTransform reconstructs embeddings from reduced dimensions
func (p *PCAReducer) InverseTransform(ctx context.Context, reduced [][]float32) ([][]float32, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if !p.isFitted {
		return nil, fmt.Errorf("PCA not fitted yet")
	}

	result := make([][]float32, len(reduced))

	for i, red := range reduced {
		// Reconstruct by multiplying with components transpose
		reconstructed := make([]float32, p.originalDim)

		// Add back the mean
		for j := 0; j < p.originalDim; j++ {
			reconstructed[j] = float32(p.mean[j])
		}

		// Add contribution from each component
		for j := 0; j < p.reducedDim; j++ {
			for k := 0; k < p.originalDim; k++ {
				reconstructed[k] += red[j] * float32(p.components[j][k])
			}
		}

		result[i] = reconstructed
	}

	return result, nil
}

// OriginalDim returns the original embedding dimension
func (p *PCAReducer) OriginalDim() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.originalDim
}

// ReducedDim returns the reduced embedding dimension
func (p *PCAReducer) ReducedDim() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.reducedDim
}

// ExplainedVarianceRatio returns variance explained by each component
func (p *PCAReducer) ExplainedVarianceRatio() []float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.explainedVariance) > p.reducedDim {
		return p.explainedVariance[:p.reducedDim]
	}
	return p.explainedVariance
}

// determineNumComponents determines optimal number of components
func (p *PCAReducer) determineNumComponents() int {
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

// truncatedSVD performs truncated Singular Value Decomposition
// This is more efficient than full eigen decomposition for large matrices
func (p *PCAReducer) truncatedSVD(ctx context.Context, centered [][]float64) ([][]float64, []float64, error) {
	n := len(centered)
	d := len(centered[0])

	// For demonstration, we'll use a simplified power iteration method
	// In production, use a proper linear algebra library like gonum

	// Determine max components to compute
	maxComponents := p.config.TargetDim
	if maxComponents == 0 || maxComponents > d {
		maxComponents = d
	}
	if maxComponents > 100 { // Limit for efficiency
		maxComponents = 100
	}

	components := make([][]float64, maxComponents)
	singularValues := make([]float64, maxComponents)

	// Power iteration for top k components
	for k := 0; k < maxComponents; k++ {
		// Check context
		select {
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		default:
		}

		// Initialize random vector
		v := make([]float64, d)
		for i := range v {
			v[i] = float64(i+k+1) / float64(d) // Deterministic initialization
		}
		normalize(v)

		// Power iteration
		maxIter := 50
		for iter := 0; iter < maxIter; iter++ {
			// v_new = A^T * A * v
			vNew := make([]float64, d)

			// First: A * v
			Av := make([]float64, n)
			for i := 0; i < n; i++ {
				for j := 0; j < d; j++ {
					Av[i] += centered[i][j] * v[j]
				}
			}

			// Then: A^T * (A * v)
			for i := 0; i < d; i++ {
				for j := 0; j < n; j++ {
					vNew[i] += centered[j][i] * Av[j]
				}
			}

			// Orthogonalize against previous components
			for i := 0; i < k; i++ {
				dot := dotProduct(vNew, components[i])
				for j := 0; j < d; j++ {
					vNew[j] -= dot * components[i][j]
				}
			}

			// Normalize
			norm := normalize(vNew)

			// Check convergence
			converged := true
			for i := range v {
				if math.Abs(v[i]-vNew[i]) > 1e-6 {
					converged = false
					break
				}
			}

			v = vNew

			if converged {
				singularValues[k] = math.Sqrt(norm / float64(n-1))
				break
			}
		}

		components[k] = v
	}

	return components, singularValues, nil
}

// Helper functions
func normalize(v []float64) float64 {
	norm := 0.0
	for _, val := range v {
		norm += val * val
	}
	norm = math.Sqrt(norm)

	if norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
	return norm
}

func dotProduct(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// GetReconstructionError calculates the reconstruction error for quality monitoring
func (p *PCAReducer) GetReconstructionError(ctx context.Context, original [][]float32) (float64, error) {
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
	for i := range original {
		for j := range original[i] {
			diff := float64(original[i][j] - reconstructed[i][j])
			totalError += diff * diff
		}
	}

	return totalError / float64(len(original)*len(original[0])), nil
}

// ExportComponents exports the PCA components for visualization or analysis
func (p *PCAReducer) ExportComponents() ComponentsInfo {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return ComponentsInfo{
		Mean:              p.mean,
		Components:        p.components,
		ExplainedVariance: p.explainedVariance[:p.reducedDim],
		OriginalDim:       p.originalDim,
		ReducedDim:        p.reducedDim,
	}
}

// ComponentsInfo contains exportable PCA information
type ComponentsInfo struct {
	Mean              []float64
	Components        [][]float64
	ExplainedVariance []float64
	OriginalDim       int
	ReducedDim        int
}

// GetTopFeatures returns the most important features for each principal component
func (p *PCAReducer) GetTopFeatures(componentIdx int, topK int) []FeatureImportance {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if componentIdx >= len(p.components) {
		return nil
	}

	component := p.components[componentIdx]
	features := make([]FeatureImportance, len(component))

	for i, weight := range component {
		features[i] = FeatureImportance{
			Index:     i,
			Weight:    weight,
			AbsWeight: math.Abs(weight),
		}
	}

	// Sort by absolute weight
	sort.Slice(features, func(i, j int) bool {
		return features[i].AbsWeight > features[j].AbsWeight
	})

	if topK > len(features) {
		topK = len(features)
	}

	return features[:topK]
}

// FeatureImportance represents the importance of a feature in a component
type FeatureImportance struct {
	Index     int
	Weight    float64
	AbsWeight float64
}
