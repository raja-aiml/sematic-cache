package reduction

import (
	"context"
	"testing"
	"time"
)

func TestNewDimensionReducer(t *testing.T) {
	config := &Config{
		TargetDim: 10,
	}

	reducer, err := NewDimensionReducer(config)
	if err != nil {
		t.Fatalf("NewDimensionReducer failed: %v", err)
	}

	if reducer == nil {
		t.Fatal("NewDimensionReducer returned nil")
	}

	if reducer.config != config {
		t.Error("config not set correctly")
	}
}

func TestDimensionReducer_Learn(t *testing.T) {
	embeddings := generateTestEmbeddings(50, 20)

	config := &Config{
		TargetDim: 5,
	}

	reducer, _ := NewDimensionReducer(config)
	ctx := context.Background()

	err := reducer.Learn(ctx, embeddings)
	if err != nil {
		t.Fatalf("Learn failed: %v", err)
	}

	if !reducer.isLearned {
		t.Error("reducer should be learned after Learn()")
	}

	if reducer.originalDim != 20 {
		t.Errorf("originalDim = %d, want 20", reducer.originalDim)
	}

	if reducer.reducedDim != 5 {
		t.Errorf("reducedDim = %d, want 5", reducer.reducedDim)
	}
}

func TestDimensionReducer_ReduceForSearch(t *testing.T) {
	embeddings := generateTestEmbeddings(50, 20)
	config := &Config{
		TargetDim: 5,
	}

	reducer, _ := NewDimensionReducer(config)
	ctx := context.Background()

	// Try to reduce before learning
	_, err := reducer.ReduceForSearch(ctx, embeddings[0])
	if err == nil {
		t.Error("ReduceForSearch should fail before learning")
	}

	// Learn
	if err := reducer.Learn(ctx, embeddings); err != nil {
		t.Fatalf("Learn failed: %v", err)
	}

	// Now reduce should work
	reduced, err := reducer.ReduceForSearch(ctx, embeddings[0])
	if err != nil {
		t.Fatalf("ReduceForSearch failed after learning: %v", err)
	}

	if len(reduced) != 5 {
		t.Errorf("Reduced dimension = %d, want 5", len(reduced))
	}
}

func TestDimensionReducer_HybridSearch(t *testing.T) {
	// Create test data
	embeddings := generateTestEmbeddings(100, 20)
	candidates := make([]SearchCandidate, len(embeddings))

	for i, emb := range embeddings {
		candidates[i] = SearchCandidate{
			ID:        string(rune('A' + i)),
			Embedding: emb,
			Metadata:  map[string]interface{}{"index": i},
		}
	}

	config := &Config{
		TargetDim: 5,
	}

	reducer, _ := NewDimensionReducer(config)
	ctx := context.Background()

	// Learn
	if err := reducer.Learn(ctx, embeddings); err != nil {
		t.Fatalf("Learn failed: %v", err)
	}

	// Generate reduced embeddings for candidates
	for i := range candidates {
		if reduced, err := reducer.ReduceForSearch(ctx, candidates[i].Embedding); err == nil {
			candidates[i].ReducedEmbedding = reduced
		}
	}

	// Test hybrid search
	queryEmbedding := embeddings[0] // Use first embedding as query
	topK := 5

	results, err := reducer.HybridSearch(ctx, queryEmbedding, candidates, topK, cosineSimilarity)
	if err != nil {
		t.Fatalf("HybridSearch failed: %v", err)
	}

	if len(results) != topK {
		t.Errorf("HybridSearch returned %d results, want %d", len(results), topK)
	}

	// First result should be the query itself with similarity ~1.0
	if len(results) > 0 {
		if results[0].Candidate.ID != "A" {
			t.Errorf("First result ID = %s, want A", results[0].Candidate.ID)
		}
		if results[0].Similarity < 0.99 {
			t.Errorf("First result similarity = %.3f, want ~1.0", results[0].Similarity)
		}
	}

	// Results should be sorted by similarity
	for i := 1; i < len(results); i++ {
		if results[i].Similarity > results[i-1].Similarity {
			t.Error("Results not sorted by similarity")
			break
		}
	}
}

func TestDimensionReducer_Metrics(t *testing.T) {
	embeddings := generateTestEmbeddings(50, 20)
	config := &Config{
		TargetDim: 5,
	}

	reducer, _ := NewDimensionReducer(config)
	ctx := context.Background()

	// Learn
	if err := reducer.Learn(ctx, embeddings); err != nil {
		t.Fatalf("Learn failed: %v", err)
	}

	// Perform some operations to generate metrics
	for i := 0; i < 10; i++ {
		_, _ = reducer.ReduceForSearch(ctx, embeddings[i])
	}

	metrics := reducer.GetMetrics()

	if metrics.ReducedDimQueries != 10 {
		t.Errorf("ReducedDimQueries = %d, want 10", metrics.ReducedDimQueries)
	}

	// Skip timing check as operations might be too fast to measure
	// Just ensure it's non-negative
	if metrics.AvgReductionTimeMs < 0 {
		t.Error("AvgReductionTimeMs should be non-negative")
	}
}

func TestDimensionReducer_GetReductionInfo(t *testing.T) {
	embeddings := generateTestEmbeddings(50, 20)
	config := &Config{
		TargetDim: 5,
	}

	reducer, _ := NewDimensionReducer(config)
	ctx := context.Background()

	// Before learning
	info := reducer.GetReductionInfo()
	if info.IsLearned {
		t.Error("IsLearned should be false before learning")
	}

	// After learning
	if err := reducer.Learn(ctx, embeddings); err != nil {
		t.Fatalf("Learn failed: %v", err)
	}

	info = reducer.GetReductionInfo()
	if !info.IsLearned {
		t.Error("IsLearned should be true after learning")
	}

	if info.OriginalDim != 20 {
		t.Errorf("OriginalDim = %d, want 20", info.OriginalDim)
	}

	if info.ReducedDim != 5 {
		t.Errorf("ReducedDim = %d, want 5", info.ReducedDim)
	}

	expectedRatio := 5.0 / 20.0
	if info.CompressionRatio != expectedRatio {
		t.Errorf("CompressionRatio = %.2f, want %.2f", info.CompressionRatio, expectedRatio)
	}
}

func TestDimensionReducer_ShouldUseReduction(t *testing.T) {
	embeddings := generateTestEmbeddings(50, 20)

	tests := []struct {
		name          string
		config        *Config
		updateMetrics func(*DimensionReducer)
		want          bool
	}{
		{
			name: "not_learned",
			config: &Config{
				TargetDim: 10,
			},
			updateMetrics: func(dr *DimensionReducer) {},
			want:          false,
		},
		{
			name: "good_compression_and_variance",
			config: &Config{
				TargetDim: 5,
			},
			updateMetrics: func(dr *DimensionReducer) {
				dr.UpdateHitRates(0.8, 0.75) // Small drop in hit rate
			},
			want: true,
		},
		{
			name: "poor_accuracy",
			config: &Config{
				TargetDim: 5,
			},
			updateMetrics: func(dr *DimensionReducer) {
				dr.UpdateHitRates(0.8, 0.6) // Large drop in hit rate
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reducer, _ := NewDimensionReducer(tt.config)
			ctx := context.Background()

			if tt.name != "not_learned" {
				if err := reducer.Learn(ctx, embeddings); err != nil {
					t.Fatalf("Learn failed: %v", err)
				}
			}

			tt.updateMetrics(reducer)

			got := reducer.ShouldUseReduction()
			if got != tt.want {
				t.Errorf("ShouldUseReduction() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDimensionReducer_EstimateSearchSpeedup(t *testing.T) {
	tests := []struct {
		name        string
		originalDim int
		reducedDim  int
		minSpeedup  float64
	}{
		{"50_percent_reduction", 100, 50, 1.5},
		{"75_percent_reduction", 100, 25, 2.5},
		{"90_percent_reduction", 100, 10, 5.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embeddings := generateTestEmbeddings(50, tt.originalDim)
			config := &Config{
				TargetDim: tt.reducedDim,
			}

			reducer, _ := NewDimensionReducer(config)
			ctx := context.Background()

			if err := reducer.Learn(ctx, embeddings); err != nil {
				t.Fatalf("Learn failed: %v", err)
			}

			speedup := reducer.EstimateSearchSpeedup()
			if speedup < tt.minSpeedup {
				t.Errorf("EstimateSearchSpeedup() = %.2f, want >= %.2f", speedup, tt.minSpeedup)
			}
		})
	}
}

func TestDimensionReducer_ReduceBatch(t *testing.T) {
	embeddings := generateTestEmbeddings(50, 20)
	config := &Config{
		TargetDim: 5,
	}

	reducer, _ := NewDimensionReducer(config)
	ctx := context.Background()

	// Learn
	if err := reducer.Learn(ctx, embeddings); err != nil {
		t.Fatalf("Learn failed: %v", err)
	}

	// Reduce batch
	batchSize := 10
	batch := embeddings[:batchSize]

	reduced, err := reducer.ReduceBatch(ctx, batch)
	if err != nil {
		t.Fatalf("ReduceBatch failed: %v", err)
	}

	if len(reduced) != batchSize {
		t.Errorf("ReduceBatch returned %d embeddings, want %d", len(reduced), batchSize)
	}

	for i, r := range reduced {
		if len(r) != 5 {
			t.Errorf("Reduced embedding %d has dimension %d, want 5", i, len(r))
		}
	}
}

func TestDimensionReducer_ConcurrentOperations(t *testing.T) {
	embeddings := generateTestEmbeddings(100, 50)
	config := &Config{
		TargetDim: 10,
	}

	reducer, _ := NewDimensionReducer(config)
	ctx := context.Background()

	// Learn
	if err := reducer.Learn(ctx, embeddings); err != nil {
		t.Fatalf("Learn failed: %v", err)
	}

	// Concurrent reduces
	done := make(chan bool, 20)

	// 10 single reduces
	for i := 0; i < 10; i++ {
		go func(idx int) {
			_, err := reducer.ReduceForSearch(ctx, embeddings[idx])
			if err != nil {
				t.Errorf("Concurrent reduce failed: %v", err)
			}
			done <- true
		}(i)
	}

	// 10 batch reduces
	for i := 0; i < 10; i++ {
		go func(start int) {
			batch := embeddings[start : start+5]
			_, err := reducer.ReduceBatch(ctx, batch)
			if err != nil {
				t.Errorf("Concurrent batch reduce failed: %v", err)
			}
			done <- true
		}(i * 5)
	}

	// Wait for all
	for i := 0; i < 20; i++ {
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("Timeout waiting for concurrent operations")
		}
	}
}

// Helper functions
func generateTestEmbeddings(n, dim int) [][]float32 {
	embeddings := make([][]float32, n)
	for i := range embeddings {
		embeddings[i] = make([]float32, dim)
		for j := range embeddings[i] {
			// Create some structure in the data
			embeddings[i][j] = float32(i+j) / float32(n+dim)
		}
	}
	return embeddings
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (sqrtFloat64(normA) * sqrtFloat64(normB))
}

func sqrtFloat64(x float64) float64 {
	if x < 0 {
		return 0
	}
	// Simple square root approximation
	z := x
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}
