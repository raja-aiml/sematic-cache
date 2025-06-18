package reduction

import (
	"context"
	"fmt"
	"math"
	"sync"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// IncrementalPCAReducer implements incremental PCA for online learning
type IncrementalPCAReducer struct {
	mu                sync.RWMutex
	config            *Config
	batchSize         int
	nSamplesSeen      int
	
	// Incremental statistics
	mean              []float64
	components        *mat.Dense
	explainedVariance []float64
	singularValues    []float64
	
	// Running sums for incremental updates
	sumOfSquares      *mat.Dense
	nComponentsSeen   int
	
	// Dimensions
	originalDim int
	reducedDim  int
	
	// State
	isInitialized bool
}

// NewIncrementalPCAReducer creates a new incremental PCA reducer
func NewIncrementalPCAReducer(config *Config, batchSize int) *IncrementalPCAReducer {
	if batchSize <= 0 {
		batchSize = 100
	}
	
	return &IncrementalPCAReducer{
		config:    config,
		batchSize: batchSize,
	}
}

// PartialFit performs incremental PCA update on a batch of samples
func (p *IncrementalPCAReducer) PartialFit(ctx context.Context, embeddings [][]float32) error {
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

	// Validate dimensions
	for i, emb := range embeddings {
		if len(emb) != d {
			return fmt.Errorf("inconsistent dimensions: embedding %d has %d dims, expected %d", i, len(emb), d)
		}
	}

	// Initialize on first batch
	if !p.isInitialized {
		return p.initializeFromBatch(ctx, embeddings)
	}

	// Validate dimension consistency
	if d != p.originalDim {
		return fmt.Errorf("dimension mismatch: got %d, expected %d", d, p.originalDim)
	}

	// Convert to matrix
	X := mat.NewDense(n, d, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			X.Set(i, j, float64(embeddings[i][j]))
		}
	}

	// Update mean incrementally
	oldNSamples := p.nSamplesSeen
	p.nSamplesSeen += n

	// Compute batch mean
	batchMean := make([]float64, d)
	for j := 0; j < d; j++ {
		col := mat.Col(nil, j, X)
		batchMean[j] = stat.Mean(col, nil)
	}

	// Update global mean
	for j := 0; j < d; j++ {
		p.mean[j] = (float64(oldNSamples)*p.mean[j] + float64(n)*batchMean[j]) / float64(p.nSamplesSeen)
	}

	// Center the batch
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			X.Set(i, j, X.At(i, j)-p.mean[j])
		}
	}

	// Update components using incremental SVD
	return p.updateComponentsIncremental(ctx, X, n)
}

// initializeFromBatch initializes the model from the first batch
func (p *IncrementalPCAReducer) initializeFromBatch(_ context.Context, embeddings [][]float32) error {
	n := len(embeddings)
	d := len(embeddings[0])
	
	p.originalDim = d
	p.nSamplesSeen = n

	// Initialize mean
	p.mean = make([]float64, d)
	
	// Convert to matrix and compute mean
	X := mat.NewDense(n, d, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			X.Set(i, j, float64(embeddings[i][j]))
			p.mean[j] += float64(embeddings[i][j])
		}
	}
	
	for j := 0; j < d; j++ {
		p.mean[j] /= float64(n)
	}

	// Center the data
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			X.Set(i, j, X.At(i, j)-p.mean[j])
		}
	}

	// Perform initial SVD
	var svd mat.SVD
	ok := svd.Factorize(X, mat.SVDThin)
	if !ok {
		return fmt.Errorf("initial SVD factorization failed")
	}

	// Get singular values
	p.singularValues = svd.Values(nil)

	// Calculate explained variance
	totalVariance := 0.0
	variances := make([]float64, len(p.singularValues))
	for i, s := range p.singularValues {
		variances[i] = (s * s) / float64(n-1)
		totalVariance += variances[i]
	}

	// Calculate explained variance ratio
	p.explainedVariance = make([]float64, len(variances))
	for i, v := range variances {
		p.explainedVariance[i] = v / totalVariance
	}

	// Determine number of components
	p.reducedDim = p.determineNumComponents()
	if p.reducedDim > len(p.singularValues) {
		p.reducedDim = len(p.singularValues)
	}

	// Get components
	vt := new(mat.Dense)
	svd.VTo(vt)
	
	p.components = mat.NewDense(p.reducedDim, d, nil)
	for i := 0; i < p.reducedDim; i++ {
		for j := 0; j < d; j++ {
			p.components.Set(i, j, vt.At(j, i))
		}
	}

	// Initialize sum of squares for future updates
	p.sumOfSquares = mat.NewDense(d, d, nil)
	p.sumOfSquares.Mul(X.T(), X)
	
	p.isInitialized = true
	p.nComponentsSeen = p.reducedDim

	return nil
}

// updateComponentsIncremental updates PCA components incrementally
func (p *IncrementalPCAReducer) updateComponentsIncremental(_ context.Context, X *mat.Dense, _ int) error {
	_, d := X.Dims()
	
	// Update sum of squares
	batchSS := mat.NewDense(d, d, nil)
	batchSS.Mul(X.T(), X)
	p.sumOfSquares.Add(p.sumOfSquares, batchSS)

	// Compute covariance matrix
	covMatrix := mat.NewDense(d, d, nil)
	covMatrix.Scale(1.0/float64(p.nSamplesSeen-1), p.sumOfSquares)

	// Perform eigendecomposition
	var eig mat.Eigen
	ok := eig.Factorize(covMatrix, mat.EigenRight)
	if !ok {
		return fmt.Errorf("eigendecomposition failed")
	}

	// Get eigenvalues and eigenvectors
	values := eig.Values(nil)
	vectors := new(mat.CDense)
	eig.VectorsTo(vectors)

	// Sort by eigenvalue magnitude (descending)
	type eigenPair struct {
		value  float64
		vector []float64
	}
	
	pairs := make([]eigenPair, d)
	for i := 0; i < d; i++ {
		realPart := real(values[i])
		vec := make([]float64, d)
		for j := 0; j < d; j++ {
			vec[j] = real(vectors.At(j, i))
		}
		pairs[i] = eigenPair{value: realPart, vector: vec}
	}

	// Sort by eigenvalue descending
	for i := 0; i < len(pairs)-1; i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[i].value < pairs[j].value {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}

	// Update components
	p.reducedDim = p.determineNumComponents()
	if p.reducedDim > d {
		p.reducedDim = d
	}

	p.components = mat.NewDense(p.reducedDim, d, nil)
	p.singularValues = make([]float64, p.reducedDim)
	p.explainedVariance = make([]float64, p.reducedDim)
	
	totalVariance := 0.0
	for i := 0; i < d && i < len(pairs); i++ {
		if i < len(pairs) && pairs[i].value > 0 {
			totalVariance += pairs[i].value
		}
	}

	for i := 0; i < p.reducedDim; i++ {
		// Set component
		for j := 0; j < d; j++ {
			p.components.Set(i, j, pairs[i].vector[j])
		}
		
		// Update singular values and explained variance
		p.singularValues[i] = math.Sqrt(pairs[i].value * float64(p.nSamplesSeen-1))
		if totalVariance > 0 {
			p.explainedVariance[i] = pairs[i].value / totalVariance
		}
	}

	incrementalUpdatesTotal.Inc()
	return nil
}

// Transform reduces dimensionality of embeddings
func (p *IncrementalPCAReducer) Transform(ctx context.Context, embeddings [][]float32) ([][]float32, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if !p.isInitialized {
		return nil, fmt.Errorf("incremental PCA not initialized")
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

// FitTransform fits the model and transforms in one step
func (p *IncrementalPCAReducer) FitTransform(ctx context.Context, embeddings [][]float32) ([][]float32, error) {
	if err := p.PartialFit(ctx, embeddings); err != nil {
		return nil, err
	}
	return p.Transform(ctx, embeddings)
}

// InverseTransform reconstructs embeddings from reduced dimensions
func (p *IncrementalPCAReducer) InverseTransform(ctx context.Context, reduced [][]float32) ([][]float32, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if !p.isInitialized {
		return nil, fmt.Errorf("incremental PCA not initialized")
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

// determineNumComponents determines optimal number of components
func (p *IncrementalPCAReducer) determineNumComponents() int {
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

// GetNSamplesSeen returns the number of samples seen so far
func (p *IncrementalPCAReducer) GetNSamplesSeen() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.nSamplesSeen
}

// IsInitialized returns whether the model has been initialized
func (p *IncrementalPCAReducer) IsInitialized() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.isInitialized
}

// OriginalDim returns the original embedding dimension
func (p *IncrementalPCAReducer) OriginalDim() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.originalDim
}

// ReducedDim returns the reduced embedding dimension
func (p *IncrementalPCAReducer) ReducedDim() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.reducedDim
}

// ExplainedVarianceRatio returns variance explained by each component
func (p *IncrementalPCAReducer) ExplainedVarianceRatio() []float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.explainedVariance) > p.reducedDim {
		return p.explainedVariance[:p.reducedDim]
	}
	return p.explainedVariance
}

// GetMean returns the current mean vector
func (p *IncrementalPCAReducer) GetMean() []float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	mean := make([]float64, len(p.mean))
	copy(mean, p.mean)
	return mean
}

// AdaptiveBatchSize adjusts batch size based on performance
func (p *IncrementalPCAReducer) AdaptiveBatchSize() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	// Simple adaptive strategy: increase batch size as we see more samples
	if p.nSamplesSeen < 1000 {
		return p.batchSize
	} else if p.nSamplesSeen < 10000 {
		return p.batchSize * 2
	} else {
		return p.batchSize * 4
	}
}

// ShouldUpdate determines if the model should be updated based on drift detection
func (p *IncrementalPCAReducer) ShouldUpdate(newBatch [][]float32, driftThreshold float64) (bool, float64) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	if !p.isInitialized || len(newBatch) == 0 {
		return true, 0.0
	}
	
	// Calculate mean of new batch
	d := len(newBatch[0])
	batchMean := make([]float64, d)
	for _, emb := range newBatch {
		for j, val := range emb {
			batchMean[j] += float64(val)
		}
	}
	for j := range batchMean {
		batchMean[j] /= float64(len(newBatch))
	}
	
	// Calculate drift as L2 distance between means
	drift := 0.0
	for j := 0; j < d; j++ {
		diff := batchMean[j] - p.mean[j]
		drift += diff * diff
	}
	drift = math.Sqrt(drift)
	
	return drift > driftThreshold, drift
}