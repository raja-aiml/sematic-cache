package reduction

import (
	"context"
	"testing"
)

func TestNewPCAReducer(t *testing.T) {
	config := &Config{
		TargetDim:   10,
		Standardize: true,
	}

	reducer := NewPCAReducer(config)
	if reducer == nil {
		t.Fatal("NewPCAReducer returned nil")
	}

	if reducer.config != config {
		t.Error("config not set correctly")
	}
}

func TestPCAReducer_Fit(t *testing.T) {
	tests := []struct {
		name       string
		config     *Config
		embeddings [][]float32
		wantErr    bool
	}{
		{
			name: "valid_embeddings",
			config: &Config{
				TargetDim: 2,
			},
			embeddings: [][]float32{
				{1.0, 2.0, 3.0, 4.0},
				{2.0, 3.0, 4.0, 5.0},
				{3.0, 4.0, 5.0, 6.0},
				{4.0, 5.0, 6.0, 7.0},
			},
			wantErr: false,
		},
		{
			name: "empty_embeddings",
			config: &Config{
				TargetDim: 2,
			},
			embeddings: [][]float32{},
			wantErr:    true,
		},
		{
			name: "variance_threshold",
			config: &Config{
				VarianceThreshold: 0.95,
			},
			embeddings: [][]float32{
				{1.0, 0.0, 0.0, 0.0},
				{0.0, 1.0, 0.0, 0.0},
				{0.0, 0.0, 1.0, 0.0},
				{0.0, 0.0, 0.0, 1.0},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reducer := NewPCAReducer(tt.config)
			ctx := context.Background()

			err := reducer.Fit(ctx, tt.embeddings)
			if (err != nil) != tt.wantErr {
				t.Errorf("Fit() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if reducer.originalDim != len(tt.embeddings[0]) {
					t.Errorf("originalDim = %d, want %d", reducer.originalDim, len(tt.embeddings[0]))
				}

				if reducer.reducedDim <= 0 {
					t.Error("reducedDim should be positive")
				}

				if !reducer.isFitted {
					t.Error("isFitted should be true after successful fit")
				}
			}
		})
	}
}

func TestPCAReducer_Transform(t *testing.T) {
	// Create test data
	embeddings := make([][]float32, 10)
	for i := range embeddings {
		embeddings[i] = make([]float32, 20)
		for j := range embeddings[i] {
			embeddings[i][j] = float32(i*j) + float32(i) + 0.1
		}
	}

	config := &Config{
		TargetDim: 5,
	}

	reducer := NewPCAReducer(config)
	ctx := context.Background()

	// Fit the reducer
	if err := reducer.Fit(ctx, embeddings); err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	// Transform new embeddings
	newEmbeddings := [][]float32{
		embeddings[0], // Same as training data
		{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0},
	}

	reduced, err := reducer.Transform(ctx, newEmbeddings)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	if len(reduced) != len(newEmbeddings) {
		t.Errorf("Transform returned %d embeddings, want %d", len(reduced), len(newEmbeddings))
	}

	for i, r := range reduced {
		if len(r) != reducer.reducedDim {
			t.Errorf("Reduced embedding %d has dimension %d, want %d", i, len(r), reducer.reducedDim)
		}
	}
}

func TestPCAReducer_FitTransform(t *testing.T) {
	embeddings := [][]float32{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}

	config := &Config{
		TargetDim: 2,
	}

	reducer := NewPCAReducer(config)
	ctx := context.Background()

	reduced, err := reducer.FitTransform(ctx, embeddings)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	if len(reduced) != len(embeddings) {
		t.Errorf("FitTransform returned %d embeddings, want %d", len(reduced), len(embeddings))
	}

	if reducer.ReducedDim() != 2 {
		t.Errorf("ReducedDim() = %d, want 2", reducer.ReducedDim())
	}
}

func TestPCAReducer_InverseTransform(t *testing.T) {
	// Create embeddings with clear structure
	embeddings := [][]float32{
		{1.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0},
		{0.0, 0.0, 1.0, 0.0},
		{0.0, 0.0, 0.0, 1.0},
		{0.5, 0.5, 0.0, 0.0},
		{0.0, 0.5, 0.5, 0.0},
	}

	config := &Config{
		TargetDim: 2,
	}

	reducer := NewPCAReducer(config)
	ctx := context.Background()

	// Fit and transform
	reduced, err := reducer.FitTransform(ctx, embeddings)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	// Inverse transform
	reconstructed, err := reducer.InverseTransform(ctx, reduced)
	if err != nil {
		t.Fatalf("InverseTransform failed: %v", err)
	}

	if len(reconstructed) != len(embeddings) {
		t.Fatalf("InverseTransform returned %d embeddings, want %d", len(reconstructed), len(embeddings))
	}

	// Check reconstruction error
	totalError := 0.0
	for i := range embeddings {
		for j := range embeddings[i] {
			diff := float64(embeddings[i][j] - reconstructed[i][j])
			totalError += diff * diff
		}
	}

	// With 2 components out of 4, we expect some reconstruction error
	if totalError == 0 {
		t.Error("Expected some reconstruction error with dimension reduction")
	}
}

func TestPCAReducer_ExplainedVarianceRatio(t *testing.T) {
	// Create data with different variance in each dimension
	embeddings := make([][]float32, 100)
	for i := range embeddings {
		embeddings[i] = []float32{
			float32(i) * 10.0, // High variance
			float32(i) * 1.0,  // Medium variance
			float32(i) * 0.1,  // Low variance
			0.01,              // Very low variance
		}
	}

	config := &Config{
		TargetDim: 3,
	}

	reducer := NewPCAReducer(config)
	ctx := context.Background()

	if err := reducer.Fit(ctx, embeddings); err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	varRatio := reducer.ExplainedVarianceRatio()
	if len(varRatio) != 3 {
		t.Errorf("ExplainedVarianceRatio returned %d values, want 3", len(varRatio))
	}

	// First component should explain most variance
	if len(varRatio) > 0 && varRatio[0] < 0.8 {
		t.Errorf("First component explains only %.2f variance, expected > 0.8", varRatio[0])
	}

	// Variance ratios should be decreasing
	for i := 1; i < len(varRatio); i++ {
		if varRatio[i] > varRatio[i-1] {
			t.Errorf("Variance ratios not decreasing: %v", varRatio)
			break
		}
	}
}

func TestPCAReducer_GetReconstructionError(t *testing.T) {
	embeddings := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 4.0, 6.0, 8.0},
		{3.0, 6.0, 9.0, 12.0},
		{4.0, 8.0, 12.0, 16.0},
	}

	tests := []struct {
		name      string
		targetDim int
		maxError  float64
	}{
		{"full_dimensions", 4, 50.0}, // Allow for numerical errors
		{"half_dimensions", 2, 10.0},
		{"one_dimension", 1, 20.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &Config{
				TargetDim: tt.targetDim,
			}

			reducer := NewPCAReducer(config)
			ctx := context.Background()

			if err := reducer.Fit(ctx, embeddings); err != nil {
				t.Fatalf("Fit failed: %v", err)
			}

			error, err := reducer.GetReconstructionError(ctx, embeddings)
			if err != nil {
				t.Fatalf("GetReconstructionError failed: %v", err)
			}

			if error > tt.maxError {
				t.Errorf("Reconstruction error %.4f exceeds max %.4f", error, tt.maxError)
			}
		})
	}
}

func TestPCAReducer_GetTopFeatures(t *testing.T) {
	// Create data where first dimension has high weight
	embeddings := make([][]float32, 50)
	for i := range embeddings {
		embeddings[i] = []float32{
			float32(i) * 10.0,
			float32(i) * 0.1,
			float32(i) * 0.01,
			0.001,
		}
	}

	config := &Config{
		TargetDim: 2,
	}

	reducer := NewPCAReducer(config)
	ctx := context.Background()

	if err := reducer.Fit(ctx, embeddings); err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Get top features for first component
	topFeatures := reducer.GetTopFeatures(0, 2)
	if len(topFeatures) != 2 {
		t.Errorf("GetTopFeatures returned %d features, want 2", len(topFeatures))
	}

	// First feature should have highest weight
	if len(topFeatures) > 0 && topFeatures[0].Index != 0 {
		t.Errorf("Expected first dimension to have highest weight, got index %d", topFeatures[0].Index)
	}
}

func TestPCAReducer_ThreadSafety(t *testing.T) {
	embeddings := make([][]float32, 100)
	for i := range embeddings {
		embeddings[i] = make([]float32, 50)
		for j := range embeddings[i] {
			embeddings[i][j] = float32(i + j)
		}
	}

	config := &Config{
		TargetDim: 10,
	}

	reducer := NewPCAReducer(config)
	ctx := context.Background()

	if err := reducer.Fit(ctx, embeddings); err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Run concurrent transforms
	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func() {
			_, err := reducer.Transform(ctx, embeddings[:10])
			if err != nil {
				t.Errorf("Concurrent transform failed: %v", err)
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}
}

