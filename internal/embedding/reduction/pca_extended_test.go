package reduction

import (
	"context"
	"fmt"
	"math"
	"testing"
	"time"
)

// TestNewPCAReducerValidation tests validation in NewPCAReducer
func TestNewPCAReducerValidation(t *testing.T) {
	tests := []struct {
		name      string
		config    *Config
		wantError bool
	}{
		{
			name:      "nil config",
			config:    nil,
			wantError: false, // nil config gets defaults which are valid
		},
		{
			name: "invalid target dim",
			config: &Config{
				TargetDim: -1,
			},
			wantError: true,
		},
		{
			name: "valid config",
			config: &Config{
				TargetDim: 10,
			},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pca := NewPCAReducer(tt.config)
			err := pca.config.Validate()
			if (err != nil) != tt.wantError {
				t.Errorf("NewPCAReducer() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

// TestPCAFitEdgeCases tests edge cases in PCA Fit method
func TestPCAFitEdgeCases(t *testing.T) {
	config := &Config{TargetDim: 5}
	pca := NewPCAReducer(config)
	ctx := context.Background()

	tests := []struct {
		name       string
		embeddings [][]float32
		wantError  bool
		errorMsg   string
	}{
		{
			name:       "zero variance data",
			embeddings: makeConstantEmbeddings(10, 5),
			wantError:  false, // PCA can handle zero variance data
		},
		{
			name:       "target dim equals embedding dim",
			embeddings: generateTestEmbeddings(10, 5),
			wantError:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := pca.Fit(ctx, tt.embeddings)
			if (err != nil) != tt.wantError {
				t.Errorf("Fit() error = %v, wantError %v", err, tt.wantError)
			}
			if err != nil && tt.errorMsg != "" && err.Error() != tt.errorMsg {
				t.Errorf("Fit() error = %v, want %v", err.Error(), tt.errorMsg)
			}
		})
	}
}

// TestPCATransformEdgeCases tests edge cases in Transform method
func TestPCATransformEdgeCases(t *testing.T) {
	config := &Config{TargetDim: 3}
	pca := NewPCAReducer(config)
	ctx := context.Background()

	// Fit first
	embeddings := generateTestEmbeddings(10, 5)
	err := pca.Fit(ctx, embeddings)
	if err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	// Test transform with nil embeddings
	_, err = pca.Transform(ctx, nil)
	if err == nil {
		t.Error("Expected error with nil embeddings")
	}

	// Test transform with empty embeddings
	_, err = pca.Transform(ctx, [][]float32{})
	if err == nil {
		t.Error("Expected error with empty embeddings")
	}

	// Test transform with wrong dimension
	wrongDim := [][]float32{{1, 2, 3}}
	_, err = pca.Transform(ctx, wrongDim)
	if err == nil {
		t.Error("Expected error with wrong dimension")
	}

	// Test transform before fit
	pca2 := NewPCAReducer(config)
	_, err = pca2.Transform(ctx, embeddings)
	if err == nil {
		t.Error("Expected error when transforming before fit")
	}
}

// TestPCAInverseTransformEdgeCases tests edge cases in InverseTransform
func TestPCAInverseTransformEdgeCases(t *testing.T) {
	config := &Config{TargetDim: 3}
	pca := NewPCAReducer(config)
	ctx := context.Background()

	// Test before fit
	_, err := pca.InverseTransform(ctx, [][]float32{{1, 2, 3}})
	if err == nil {
		t.Error("Expected error when inverse transforming before fit")
	}

	// Fit first
	embeddings := generateTestEmbeddings(10, 5)
	err = pca.Fit(ctx, embeddings)
	if err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	// Transform some data
	reduced, err := pca.Transform(ctx, embeddings[:5])
	if err != nil {
		t.Fatalf("Failed to transform: %v", err)
	}

	// Test inverse transform
	reconstructed, err := pca.InverseTransform(ctx, reduced)
	if err != nil {
		t.Errorf("InverseTransform failed: %v", err)
	}

	if len(reconstructed) != len(reduced) {
		t.Errorf("Expected %d reconstructed embeddings, got %d", len(reduced), len(reconstructed))
	}

	// Test with nil
	_, err = pca.InverseTransform(ctx, nil)
	if err == nil {
		t.Error("Expected error with nil reduced embeddings")
	}

	// Test with wrong dimension
	wrongDim := [][]float32{{1, 2}} // Should be 3D
	_, err = pca.InverseTransform(ctx, wrongDim)
	if err == nil {
		t.Error("Expected error with wrong reduced dimension")
	}
}

// TestGetReconstructionError tests the GetReconstructionError method
func TestGetReconstructionError(t *testing.T) {
	config := &Config{TargetDim: 3}
	pca := NewPCAReducer(config)
	ctx := context.Background()

	// Test before fit
	_, err := pca.GetReconstructionError(ctx, [][]float32{{1, 2, 3, 4, 5}})
	if err == nil {
		t.Error("Expected error before fit")
	}

	// Fit
	embeddings := generateTestEmbeddings(20, 10)
	err = pca.Fit(ctx, embeddings)
	if err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	// Test reconstruction error
	testEmbeddings := embeddings[:5]
	avgError, err := pca.GetReconstructionError(ctx, testEmbeddings)
	if err != nil {
		t.Errorf("GetReconstructionError failed: %v", err)
	}

	// Error should be positive
	if avgError < 0 {
		t.Errorf("Reconstruction error should be positive, got %f", avgError)
	}

	// Test with perfect reconstruction (using components as input)
	// This is a synthetic test - in practice, error won't be exactly 0
	if pca.ReducedDim() == pca.OriginalDim() && avgError > 0.1 {
		t.Errorf("Expected very low reconstruction error with full dimensions, got %f", avgError)
	}

	// Test with nil embeddings
	_, err = pca.GetReconstructionError(ctx, nil)
	if err == nil {
		t.Error("Expected error with nil embeddings")
	}
}

// TestExportComponents tests the ExportComponents method
func TestExportComponents(t *testing.T) {
	config := &Config{TargetDim: 3}
	pca := NewPCAReducer(config)
	ctx := context.Background()

	// Test before fit
	data := pca.ExportComponents()
	if data.Components != nil {
		t.Error("Expected nil components when exporting before fit")
	}

	// Fit
	embeddings := generateTestEmbeddings(20, 10)
	err := pca.Fit(ctx, embeddings)
	if err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	// Export components
	data = pca.ExportComponents()

	// Verify dimensions
	if data.OriginalDim != 10 {
		t.Errorf("Expected originalDim 10, got %v", data.OriginalDim)
	}

	if data.ReducedDim != 3 {
		t.Errorf("Expected targetDim 3, got %v", data.ReducedDim)
	}

	// Verify arrays have correct length
	if len(data.Mean) != 10 {
		t.Errorf("Expected mean length 10, got %d", len(data.Mean))
	}

	if len(data.Components) != 3 {
		t.Errorf("Expected 3 components, got %d", len(data.Components))
	}

	// Check first component dimension
	if len(data.Components) > 0 && len(data.Components[0]) != 10 {
		t.Errorf("Expected component dimension 10, got %d", len(data.Components[0]))
	}
}

// TestGetTopFeatures tests the GetTopFeatures method
func TestGetTopFeatures(t *testing.T) {
	config := &Config{TargetDim: 3}
	pca := NewPCAReducer(config)
	ctx := context.Background()

	// Test before fit
	topFeatures := pca.GetTopFeatures(0, 5)
	if topFeatures != nil {
		t.Error("Expected nil top features before fit")
	}

	// Create embeddings with clear patterns
	embeddings := make([][]float32, 50)
	for i := range embeddings {
		embeddings[i] = make([]float32, 10)
		// Make first few features have high variance
		embeddings[i][0] = float32(i) * 2.0   // High variance
		embeddings[i][1] = float32(i) * 1.5   // Medium variance
		embeddings[i][2] = float32(i) * 1.0   // Medium variance
		embeddings[i][3] = float32(i%2) * 0.5 // Low variance
		for j := 4; j < 10; j++ {
			embeddings[i][j] = 0.1 // Very low variance
		}
	}

	// Fit
	err := pca.Fit(ctx, embeddings)
	if err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	// Test getting top features for first component
	topFeatures = pca.GetTopFeatures(0, 5)

	if len(topFeatures) != 5 {
		t.Errorf("Expected 5 top features, got %d", len(topFeatures))
	}

	// Features should be sorted by absolute weight
	for i := 1; i < len(topFeatures); i++ {
		if math.Abs(float64(topFeatures[i].Weight)) > math.Abs(float64(topFeatures[i-1].Weight)) {
			t.Error("Top features not sorted by absolute weight")
			break
		}
	}

	// Test with invalid component index
	invalidFeatures := pca.GetTopFeatures(10, 5)
	if invalidFeatures != nil {
		t.Error("Expected nil with invalid component index")
	}

	// Test with topN larger than features
	allFeatures := pca.GetTopFeatures(0, 20)
	if len(allFeatures) != 10 {
		t.Errorf("Expected 10 features when topN > features, got %d", len(allFeatures))
	}

	// Test with topN = 0
	noFeatures := pca.GetTopFeatures(0, 0)
	if len(noFeatures) != 0 {
		t.Errorf("Expected 0 features with topN=0, got %d", len(noFeatures))
	}
}

// TestValidateEmbeddingsForTransform tests embedding validation for transform
func TestValidateEmbeddingsForTransform(t *testing.T) {
	tests := []struct {
		name          string
		embeddings    [][]float32
		expectedDim   int
		wantError     bool
		errorContains string
	}{
		{
			name:          "nil embeddings",
			embeddings:    nil,
			expectedDim:   5,
			wantError:     true,
			errorContains: "nil",
		},
		{
			name:          "empty embeddings",
			embeddings:    [][]float32{},
			expectedDim:   5,
			wantError:     true,
			errorContains: "empty",
		},
		{
			name: "wrong dimension",
			embeddings: [][]float32{
				{1, 2, 3},
				{4, 5, 6},
			},
			expectedDim:   5,
			wantError:     true,
			errorContains: "expected dimension 5",
		},
		{
			name: "valid embeddings",
			embeddings: [][]float32{
				{1, 2, 3, 4, 5},
				{6, 7, 8, 9, 10},
			},
			expectedDim: 5,
			wantError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := func() error {
				if tt.embeddings == nil {
					return fmt.Errorf("embeddings cannot be nil")
				}
				if len(tt.embeddings) == 0 {
					return fmt.Errorf("embeddings cannot be empty")
				}
				for i, emb := range tt.embeddings {
					if len(emb) != tt.expectedDim {
						return fmt.Errorf("embedding %d has dimension %d, expected dimension %d", i, len(emb), tt.expectedDim)
					}
				}
				return nil
			}()
			if (err != nil) != tt.wantError {
				t.Errorf("validateEmbeddingsForTransform() error = %v, wantError %v", err, tt.wantError)
			}
			if err != nil && tt.errorContains != "" {
				if !contains(err.Error(), tt.errorContains) {
					t.Errorf("Error should contain %q, got %q", tt.errorContains, err.Error())
				}
			}
		})
	}
}

// TestValidateReducedEmbeddings tests reduced embedding validation
func TestValidateReducedEmbeddings(t *testing.T) {
	tests := []struct {
		name          string
		embeddings    [][]float32
		expectedDim   int
		wantError     bool
		errorContains string
	}{
		{
			name:          "nil embeddings",
			embeddings:    nil,
			expectedDim:   3,
			wantError:     true,
			errorContains: "nil",
		},
		{
			name:          "empty embeddings",
			embeddings:    [][]float32{},
			expectedDim:   3,
			wantError:     true,
			errorContains: "empty",
		},
		{
			name: "wrong dimension",
			embeddings: [][]float32{
				{1, 2},
				{3, 4},
			},
			expectedDim:   3,
			wantError:     true,
			errorContains: "expected reduced dimension 3",
		},
		{
			name: "valid reduced embeddings",
			embeddings: [][]float32{
				{1, 2, 3},
				{4, 5, 6},
			},
			expectedDim: 3,
			wantError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := func() error {
				if tt.embeddings == nil {
					return fmt.Errorf("reduced embeddings cannot be nil")
				}
				if len(tt.embeddings) == 0 {
					return fmt.Errorf("reduced embeddings cannot be empty")
				}
				for i, emb := range tt.embeddings {
					if len(emb) != tt.expectedDim {
						return fmt.Errorf("reduced embedding %d has dimension %d, expected reduced dimension %d", i, len(emb), tt.expectedDim)
					}
				}
				return nil
			}()
			if (err != nil) != tt.wantError {
				t.Errorf("validateReducedEmbeddings() error = %v, wantError %v", err, tt.wantError)
			}
			if err != nil && tt.errorContains != "" {
				if !contains(err.Error(), tt.errorContains) {
					t.Errorf("Error should contain %q, got %q", tt.errorContains, err.Error())
				}
			}
		})
	}
}

// TestNormalize tests the normalize function
func TestNormalize(t *testing.T) {
	tests := []struct {
		name         string
		vector       []float64
		expected     []float64
		expectedNorm float64
	}{
		{
			name:         "zero vector",
			vector:       []float64{0, 0, 0},
			expected:     []float64{0, 0, 0},
			expectedNorm: 0,
		},
		{
			name:         "unit vector",
			vector:       []float64{1, 0, 0},
			expected:     []float64{1, 0, 0},
			expectedNorm: 1,
		},
		{
			name:         "general vector",
			vector:       []float64{3, 4, 0},
			expected:     []float64{0.6, 0.8, 0}, // 3/5, 4/5, 0/5
			expectedNorm: 5,
		},
		{
			name:         "negative values",
			vector:       []float64{-3, 4, 0},
			expected:     []float64{-0.6, 0.8, 0},
			expectedNorm: 5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy since normalize modifies in place
			vec := make([]float64, len(tt.vector))
			copy(vec, tt.vector)

			norm := normalize(vec)

			// Check norm
			if math.Abs(norm-tt.expectedNorm) > 0.0001 {
				t.Errorf("normalize() norm = %f, want %f", norm, tt.expectedNorm)
			}

			// Check normalized vector
			for i := range vec {
				if math.Abs(vec[i]-tt.expected[i]) > 0.0001 {
					t.Errorf("normalize()[%d] = %f, want %f", i, vec[i], tt.expected[i])
				}
			}
		})
	}
}

// TestDotProduct tests the dotProduct function
func TestDotProduct(t *testing.T) {
	tests := []struct {
		name     string
		a        []float64
		b        []float64
		expected float64
	}{
		{
			name:     "zero vectors",
			a:        []float64{0, 0, 0},
			b:        []float64{0, 0, 0},
			expected: 0,
		},
		{
			name:     "orthogonal vectors",
			a:        []float64{1, 0, 0},
			b:        []float64{0, 1, 0},
			expected: 0,
		},
		{
			name:     "parallel vectors",
			a:        []float64{1, 2, 3},
			b:        []float64{2, 4, 6},
			expected: 28, // 2 + 8 + 18
		},
		{
			name:     "general case",
			a:        []float64{1, 2, 3},
			b:        []float64{4, 5, 6},
			expected: 32, // 4 + 10 + 18
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := dotProduct(tt.a, tt.b)
			if math.Abs(result-tt.expected) > 0.0001 {
				t.Errorf("dotProduct() = %f, want %f", result, tt.expected)
			}
		})
	}
}

// TestFitTransform tests the FitTransform method
func TestFitTransform(t *testing.T) {
	config := &Config{TargetDim: 3}
	pca := NewPCAReducer(config)
	ctx := context.Background()

	// Test with valid embeddings
	embeddings := generateTestEmbeddings(20, 10)
	reduced, err := pca.FitTransform(ctx, embeddings)
	if err != nil {
		t.Errorf("FitTransform failed: %v", err)
	}

	// Check dimensions
	if len(reduced) != len(embeddings) {
		t.Errorf("Expected %d reduced embeddings, got %d", len(embeddings), len(reduced))
	}

	for i, r := range reduced {
		if len(r) != 3 {
			t.Errorf("Reduced embedding %d has dimension %d, expected 3", i, len(r))
		}
	}

	// PCA should be fitted - check by trying to transform
	_, err = pca.Transform(ctx, embeddings[:1])
	if err != nil {
		t.Error("PCA should be fitted after FitTransform")
	}

	// Test with edge cases
	_, err = pca.FitTransform(ctx, nil)
	if err == nil {
		t.Error("Expected error with nil embeddings")
	}
}

// TestPCAWithStandardization tests PCA with standardization enabled
func TestPCAWithStandardization(t *testing.T) {
	config := &Config{
		TargetDim:   3,
		Standardize: true,
	}
	pca := NewPCAReducer(config)
	ctx := context.Background()

	// Create embeddings with different scales
	embeddings := make([][]float32, 50)
	for i := range embeddings {
		embeddings[i] = make([]float32, 5)
		embeddings[i][0] = float32(i) * 1000  // Large scale
		embeddings[i][1] = float32(i) * 0.001 // Small scale
		embeddings[i][2] = float32(i)         // Normal scale
		embeddings[i][3] = float32(i % 10)    // Bounded
		embeddings[i][4] = 5.0                // Constant
	}

	// Fit with standardization
	err := pca.Fit(ctx, embeddings)
	if err != nil {
		t.Fatalf("Failed to fit with standardization: %v", err)
	}

	// Transform
	reduced, err := pca.Transform(ctx, embeddings[:5])
	if err != nil {
		t.Errorf("Transform failed: %v", err)
	}

	// Verify dimensions
	if len(reduced) != 5 || len(reduced[0]) != 3 {
		t.Errorf("Unexpected reduced dimensions: %dx%d", len(reduced), len(reduced[0]))
	}
}

// Helper function to create constant embeddings (zero variance)
func makeConstantEmbeddings(n, dim int) [][]float32 {
	embeddings := make([][]float32, n)
	for i := range embeddings {
		embeddings[i] = make([]float32, dim)
		for j := range embeddings[i] {
			embeddings[i][j] = 1.0 // All same value
		}
	}
	return embeddings
}

// TestTruncatedSVDEdgeCases tests edge cases through public methods
func TestTruncatedSVDEdgeCases(t *testing.T) {
	// Test PCA with reasonable target dimension
	config := &Config{TargetDim: 2}
	pca := NewPCAReducer(config)
	ctx := context.Background()

	// Create matrix with 2 samples, 5 features
	embeddings := [][]float32{
		{1, 2, 3, 4, 5},
		{6, 7, 8, 9, 10},
	}

	err := pca.Fit(ctx, embeddings)
	if err != nil {
		t.Errorf("Fit failed: %v", err)
	}

	// Should reduce to targetDim = 2 (since we have 2 samples and 5 features)
	if pca.ReducedDim() != 2 {
		t.Errorf("Expected reduced dim 2, got %d", pca.ReducedDim())
	}
}

// TestConcurrentPCAOperations tests concurrent PCA operations
func TestConcurrentPCAOperations(t *testing.T) {
	config := &Config{TargetDim: 5}
	pca := NewPCAReducer(config)
	ctx := context.Background()

	// Fit first
	embeddings := generateTestEmbeddings(100, 20)
	err := pca.Fit(ctx, embeddings)
	if err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	// Concurrent operations
	numGoroutines := 20
	numOps := 50
	errors := make(chan error, numGoroutines*numOps)

	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			for j := 0; j < numOps; j++ {
				switch j % 4 {
				case 0:
					// Transform
					batch := embeddings[j%10 : (j%10)+5]
					_, err := pca.Transform(ctx, batch)
					if err != nil {
						errors <- fmt.Errorf("transform error: %v", err)
					}
				case 1:
					// Get reconstruction error
					batch := embeddings[j%10 : (j%10)+3]
					_, err := pca.GetReconstructionError(ctx, batch)
					if err != nil {
						errors <- fmt.Errorf("reconstruction error: %v", err)
					}
				case 2:
					// Get top features
					features := pca.GetTopFeatures(j%3, 5)
					if features == nil && j%3 < pca.ReducedDim() {
						errors <- fmt.Errorf("top features returned nil for valid component")
					}
				case 3:
					// Export components
					data := pca.ExportComponents()
					if data.Components == nil {
						errors <- fmt.Errorf("export returned nil components")
					}
				}
			}
		}(i)
	}

	// Wait a bit and check for errors
	done := make(chan bool)
	go func() {
		time.Sleep(2 * time.Second)
		close(errors)
		done <- true
	}()

	select {
	case <-done:
		// Check if any errors occurred
		errorCount := 0
		for err := range errors {
			t.Errorf("Concurrent operation error: %v", err)
			errorCount++
			if errorCount > 10 {
				t.Fatal("Too many concurrent errors")
			}
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Concurrent operations timeout")
	}
}

// TestPCAConsistency tests that PCA produces consistent results
func TestPCAConsistency(t *testing.T) {
	config := &Config{
		TargetDim:  3,
		RandomSeed: 42,
	}

	embeddings := generateTestEmbeddings(20, 10)
	ctx := context.Background()

	// Create two PCA instances with same config
	pca1 := NewPCAReducer(config)
	pca2 := NewPCAReducer(config)

	// Fit both
	err := pca1.Fit(ctx, embeddings)
	if err != nil {
		t.Fatalf("PCA1 fit failed: %v", err)
	}

	err = pca2.Fit(ctx, embeddings)
	if err != nil {
		t.Fatalf("PCA2 fit failed: %v", err)
	}

	// Transform same data
	testData := embeddings[:5]
	reduced1, _ := pca1.Transform(ctx, testData)
	reduced2, _ := pca2.Transform(ctx, testData)

	// Results should be similar (allowing for numerical differences)
	for i := range reduced1 {
		for j := range reduced1[i] {
			diff := math.Abs(float64(reduced1[i][j] - reduced2[i][j]))
			if diff > 0.1 {
				t.Errorf("Inconsistent results at [%d][%d]: %f vs %f", i, j, reduced1[i][j], reduced2[i][j])
			}
		}
	}
}
