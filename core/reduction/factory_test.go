package reduction

import (
	"context"
	"math"
	"math/rand"
	"testing"
)

func TestReducerFactory(t *testing.T) {
	factory := NewReducerFactory()

	t.Run("CreatePCAReducer", func(t *testing.T) {
		config := ReducerConfig{
			OutputDimensions: 10,
			VarianceRetained: 0.95,
		}

		reducer, err := factory.CreateReducer(PCAReducerType, config)
		if err != nil {
			t.Fatalf("Failed to create PCA reducer: %v", err)
		}

		if reducer == nil {
			t.Fatal("Expected non-nil reducer")
		}

		// Type assertion to verify correct type
		if _, ok := reducer.(*PCAReducer); !ok {
			t.Fatal("Expected PCAReducer type")
		}
	})

	t.Run("CreatePCAGonumReducer", func(t *testing.T) {
		config := ReducerConfig{
			OutputDimensions: 10,
			VarianceRetained: 0.95,
		}

		reducer, err := factory.CreateReducer(PCAGonumReducerType, config)
		if err != nil {
			t.Fatalf("Failed to create PCA Gonum reducer: %v", err)
		}

		if reducer == nil {
			t.Fatal("Expected non-nil reducer")
		}

		// Type assertion to verify correct type
		if _, ok := reducer.(*PCAGonumReducer); !ok {
			t.Fatal("Expected PCAGonumReducer type")
		}
	})

	t.Run("CreateIncrementalPCAReducer", func(t *testing.T) {
		config := ReducerConfig{
			OutputDimensions: 10,
			VarianceRetained: 0.95,
		}

		reducer, err := factory.CreateReducer(IncrementalPCAReducerType, config)
		if err != nil {
			t.Fatalf("Failed to create Incremental PCA reducer: %v", err)
		}

		if reducer == nil {
			t.Fatal("Expected non-nil reducer")
		}

		// Type assertion to verify correct type
		if _, ok := reducer.(*IncrementalPCAReducer); !ok {
			t.Fatal("Expected IncrementalPCAReducer type")
		}
	})

	t.Run("InvalidReducerType", func(t *testing.T) {
		config := ReducerConfig{
			OutputDimensions: 10,
			VarianceRetained: 0.95,
		}

		_, err := factory.CreateReducer("invalid_type", config)
		if err == nil {
			t.Fatal("Expected error for invalid reducer type")
		}
	})

	t.Run("GetAvailableReducers", func(t *testing.T) {
		reducers := factory.GetAvailableReducers()
		if len(reducers) != 3 {
			t.Fatalf("Expected 3 available reducers, got %d", len(reducers))
		}

		// Check that all expected types are present
		found := make(map[ReducerType]bool)
		for _, r := range reducers {
			found[r] = true
		}

		expectedTypes := []ReducerType{
			PCAReducerType,
			PCAGonumReducerType,
			IncrementalPCAReducerType,
		}

		for _, expected := range expectedTypes {
			if !found[expected] {
				t.Errorf("Expected reducer type %s not found", expected)
			}
		}
	})
}

func TestDimensionReducerWithFactory(t *testing.T) {
	ctx := context.Background()

	// Generate test data with more variance
	embeddings := generateVariedTestEmbeddings(100, 50)

	testCases := []struct {
		name   string
		config DimensionReducerConfig
	}{
		{
			name: "StandardPCA",
			config: DimensionReducerConfig{
				ReducerConfig: ReducerConfig{
					OutputDimensions: 10,
					VarianceRetained: 0.0, // Don't use variance threshold, use fixed dimensions
				},
				Type:               PCAReducerType,
				EnableOptimization: false,
			},
		},
		{
			name: "OptimizedPCA",
			config: DimensionReducerConfig{
				ReducerConfig: ReducerConfig{
					OutputDimensions: 10,
					VarianceRetained: 0.0, // Don't use variance threshold, use fixed dimensions
				},
				Type:               PCAReducerType,
				EnableOptimization: true, // Should use Gonum
			},
		},
		{
			name: "IncrementalPCA",
			config: DimensionReducerConfig{
				ReducerConfig: ReducerConfig{
					OutputDimensions: 10,
					VarianceRetained: 0.0, // Don't use variance threshold, use fixed dimensions
				},
				Type:               IncrementalPCAReducerType,
				EnableOptimization: false,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create reducer using factory
			reducer, err := NewDimensionReducerWithFactory(tc.config)
			if err != nil {
				t.Fatalf("Failed to create dimension reducer: %v", err)
			}

			// Train the reducer
			err = reducer.Learn(ctx, embeddings)
			if err != nil {
				t.Fatalf("Failed to train reducer: %v", err)
			}

			// Get reduction info
			info := reducer.GetReductionInfo()
			if !info.IsLearned {
				t.Fatal("Expected reducer to be learned")
			}

			if info.OriginalDim != 50 {
				t.Errorf("Expected original dim 50, got %d", info.OriginalDim)
			}

			if info.ReducedDim != 10 {
				t.Errorf("Expected reduced dim 10, got %d", info.ReducedDim)
			}

			// Test transformation
			testEmb := embeddings[0]
			reduced, err := reducer.ReduceForSearch(ctx, testEmb)
			if err != nil {
				t.Fatalf("Failed to reduce embedding: %v", err)
			}

			if len(reduced) != 10 {
				t.Errorf("Expected reduced dimension 10, got %d", len(reduced))
			}

			// Test batch transformation
			batchReduced, err := reducer.ReduceBatch(ctx, embeddings[:5])
			if err != nil {
				t.Fatalf("Failed to reduce batch: %v", err)
			}

			if len(batchReduced) != 5 {
				t.Errorf("Expected 5 reduced embeddings, got %d", len(batchReduced))
			}
		})
	}
}

func TestAllReducerImplementations(t *testing.T) {
	ctx := context.Background()
	factory := NewReducerFactory()

	// Test data with more variance
	embeddings := generateVariedTestEmbeddings(50, 20)
	testEmb := embeddings[0]

	// Test each available reducer type
	for _, reducerType := range factory.GetAvailableReducers() {
		t.Run(string(reducerType), func(t *testing.T) {
			config := ReducerConfig{
				OutputDimensions: 5,
				VarianceRetained: 0.0, // Don't use variance threshold, use fixed dimensions
			}

			// Create reducer
			reducer, err := factory.CreateReducer(reducerType, config)
			if err != nil {
				t.Fatalf("Failed to create reducer: %v", err)
			}

			// Test Fit
			err = reducer.Fit(ctx, embeddings)
			if err != nil {
				t.Fatalf("Failed to fit reducer: %v", err)
			}

			// Test dimensions
			if reducer.OriginalDim() != 20 {
				t.Errorf("Expected original dim 20, got %d", reducer.OriginalDim())
			}

			if reducer.ReducedDim() != 5 {
				t.Errorf("Expected reduced dim 5, got %d", reducer.ReducedDim())
			}

			// Test Transform
			transformed, err := reducer.Transform(ctx, [][]float32{testEmb})
			if err != nil {
				t.Fatalf("Failed to transform: %v", err)
			}

			if len(transformed) != 1 {
				t.Errorf("Expected 1 transformed embedding, got %d", len(transformed))
			}

			if len(transformed[0]) != 5 {
				t.Errorf("Expected dimension 5, got %d", len(transformed[0]))
			}

			// Test ExplainedVarianceRatio
			varRatio := reducer.ExplainedVarianceRatio()
			if len(varRatio) != 5 {
				t.Errorf("Expected 5 variance ratios, got %d", len(varRatio))
			}

			// Verify variance ratios sum to <= 1
			sum := 0.0
			for _, v := range varRatio {
				sum += v
			}
			if sum > 1.01 { // Allow small numerical error
				t.Errorf("Variance ratios sum to %f, expected <= 1", sum)
			}
		})
	}
}

// generateVariedTestEmbeddings generates test embeddings with more variance
func generateVariedTestEmbeddings(count, dim int) [][]float32 {
	embeddings := make([][]float32, count)
	for i := 0; i < count; i++ {
		embedding := make([]float32, dim)
		for j := 0; j < dim; j++ {
			// Create embeddings with multiple patterns to ensure variance
			base := float32(j) * 0.1
			// Add different patterns for different dimensions
			if j < dim/4 {
				// Linear pattern
				embedding[j] = base + float32(i)*0.05
			} else if j < dim/2 {
				// Quadratic pattern
				embedding[j] = base + float32(i*i)*0.001
			} else if j < 3*dim/4 {
				// Sinusoidal pattern
				embedding[j] = base + float32(math.Sin(float64(i)*0.1))*0.5
			} else {
				// Random noise
				embedding[j] = base + float32(rand.Float64())*0.2
			}
		}
		embeddings[i] = embedding
	}
	return embeddings
}