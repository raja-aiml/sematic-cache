package reduction

import (
	"context"
	"testing"
)

// TestGonumPCA tests the Gonum-based PCA implementation
func TestGonumPCA(t *testing.T) {
	config := &Config{
		TargetDim:         2,
		VarianceThreshold: 0,
		Standardize:       false,
	}

	pca := NewPCAGonumReducer(config)
	ctx := context.Background()

	// Create simple test data
	embeddings := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 3.0, 4.0, 5.0},
		{3.0, 4.0, 5.0, 6.0},
		{4.0, 5.0, 6.0, 7.0},
		{5.0, 6.0, 7.0, 8.0},
	}

	// Fit the model
	err := pca.Fit(ctx, embeddings)
	if err != nil {
		t.Fatalf("Failed to fit PCA: %v", err)
	}

	// Check dimensions
	if pca.OriginalDim() != 4 {
		t.Errorf("Expected original dim 4, got %d", pca.OriginalDim())
	}
	if pca.ReducedDim() != 2 {
		t.Errorf("Expected reduced dim 2, got %d", pca.ReducedDim())
	}

	// Transform data
	reduced, err := pca.Transform(ctx, embeddings)
	if err != nil {
		t.Fatalf("Failed to transform: %v", err)
	}

	if len(reduced) != len(embeddings) {
		t.Errorf("Expected %d reduced embeddings, got %d", len(embeddings), len(reduced))
	}

	for i, r := range reduced {
		if len(r) != 2 {
			t.Errorf("Expected reduced dim 2 for embedding %d, got %d", i, len(r))
		}
	}

	// Test reconstruction
	reconstructed, err := pca.InverseTransform(ctx, reduced)
	if err != nil {
		t.Fatalf("Failed to inverse transform: %v", err)
	}

	if len(reconstructed) != len(embeddings) {
		t.Errorf("Expected %d reconstructed embeddings, got %d", len(embeddings), len(reconstructed))
	}

	// Test batch transform
	batchReduced, err := pca.BatchTransform(ctx, embeddings, 2)
	if err != nil {
		t.Fatalf("Failed to batch transform: %v", err)
	}

	if len(batchReduced) != len(embeddings) {
		t.Errorf("Expected %d batch reduced embeddings, got %d", len(embeddings), len(batchReduced))
	}
}

// TestHeapTopK tests the heap-based top-K selection
func TestHeapTopK(t *testing.T) {
	// Create test candidates
	candidates := []scoredCandidate{
		{candidate: SearchCandidate{ID: "1"}, similarity: 0.1},
		{candidate: SearchCandidate{ID: "2"}, similarity: 0.9},
		{candidate: SearchCandidate{ID: "3"}, similarity: 0.5},
		{candidate: SearchCandidate{ID: "4"}, similarity: 0.7},
		{candidate: SearchCandidate{ID: "5"}, similarity: 0.3},
		{candidate: SearchCandidate{ID: "6"}, similarity: 0.8},
		{candidate: SearchCandidate{ID: "7"}, similarity: 0.2},
		{candidate: SearchCandidate{ID: "8"}, similarity: 0.6},
		{candidate: SearchCandidate{ID: "9"}, similarity: 0.4},
		{candidate: SearchCandidate{ID: "10"}, similarity: 0.95},
	}

	selector := NewTopKSelector(3)
	results := selector.SelectTopK(candidates)

	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}

	// Check that we got the top 3 by similarity
	expectedIDs := []string{"10", "2", "6"}
	for i, result := range results {
		if result.ID != expectedIDs[i] {
			t.Errorf("Expected ID %s at position %d, got %s", expectedIDs[i], i, result.ID)
		}
	}
}

// TestObjectPooling tests the object pooling functionality
func TestObjectPooling(t *testing.T) {
	// Test embedding pool
	emb1 := GetEmbedding(384)
	if cap(emb1) < 384 {
		t.Errorf("Expected embedding capacity >= 384, got %d", cap(emb1))
	}
	PutEmbedding(emb1)

	emb2 := GetEmbedding(384)
	// Should get the same underlying array (pooled)
	if cap(emb2) != cap(emb1) {
		t.Log("Warning: Pool might not be reusing embeddings efficiently")
	}

	// Test candidate pool
	cand1 := GetCandidate()
	if cand1.Metadata == nil {
		t.Error("Expected candidate to have initialized metadata map")
	}
	cand1.ID = "test"
	PutCandidate(cand1)

	cand2 := GetCandidate()
	if cand2.ID != "" {
		t.Error("Expected pooled candidate to be reset")
	}

	// Test slice pools
	candSlice := GetCandidateSlice(100)
	if cap(candSlice) < 100 {
		t.Errorf("Expected candidate slice capacity >= 100, got %d", cap(candSlice))
	}
	PutCandidateSlice(candSlice)

	// Test power of 2 rounding
	n := nextPowerOf2(100)
	if n != 128 {
		t.Errorf("Expected nextPowerOf2(100) = 128, got %d", n)
	}
}

// TestOptimizedReducer tests the optimized dimension reducer
func TestOptimizedReducer(t *testing.T) {
	config := &Config{
		TargetDim:         2,
		VarianceThreshold: 0,
		Standardize:       false,
	}

	reducer, err := NewOptimizedDimensionReducer(config)
	if err != nil {
		t.Fatalf("Failed to create optimized reducer: %v", err)
	}

	ctx := context.Background()

	// Create test data
	embeddings := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 3.0, 4.0, 5.0},
		{3.0, 4.0, 5.0, 6.0},
		{4.0, 5.0, 6.0, 7.0},
		{5.0, 6.0, 7.0, 8.0},
	}

	// Learn
	err = reducer.Learn(ctx, embeddings)
	if err != nil {
		t.Fatalf("Failed to learn: %v", err)
	}

	// Create candidates
	reduced, _ := reducer.ReduceBatch(ctx, embeddings)
	candidates := make([]SearchCandidate, len(embeddings))
	for i := range embeddings {
		candidates[i] = SearchCandidate{
			ID:               string(rune('A' + i)),
			Embedding:        embeddings[i],
			ReducedEmbedding: reduced[i],
		}
	}

	// Test optimized hybrid search
	query := []float32{3.0, 4.0, 5.0, 6.0}
	results, err := reducer.OptimizedHybridSearch(ctx, query, candidates, 3, func(a, b []float32) float64 {
		// Simple dot product
		var sum float64
		for i := range a {
			sum += float64(a[i]) * float64(b[i])
		}
		return sum
	})

	if err != nil {
		t.Fatalf("Failed hybrid search: %v", err)
	}

	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}

	// Test with object pooling disabled
	reducer.EnableObjectPooling(false)
	results2, err := reducer.OptimizedHybridSearch(ctx, query, candidates, 3, func(a, b []float32) float64 {
		var sum float64
		for i := range a {
			sum += float64(a[i]) * float64(b[i])
		}
		return sum
	})

	if err != nil {
		t.Fatalf("Failed hybrid search without pooling: %v", err)
	}

	if len(results2) != len(results) {
		t.Error("Results differ with/without pooling")
	}
}
