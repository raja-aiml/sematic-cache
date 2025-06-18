package reduction

import (
	"context"
	"math"
	"testing"
)

func TestPCAReducer_ValidationErrors(t *testing.T) {
	ctx := context.Background()

	t.Run("nil reducer", func(t *testing.T) {
		var p *PCAReducer

		// Test all methods with nil reducer
		err := p.Fit(ctx, [][]float32{{1, 2, 3}})
		if err == nil || err.Error() != "PCAReducer is nil" {
			t.Errorf("expected nil PCAReducer error, got %v", err)
		}

		_, err = p.Transform(ctx, [][]float32{{1, 2, 3}})
		if err == nil || err.Error() != "PCAReducer is nil" {
			t.Errorf("expected nil PCAReducer error, got %v", err)
		}

		_, err = p.FitTransform(ctx, [][]float32{{1, 2, 3}})
		if err == nil || err.Error() != "PCAReducer is nil" {
			t.Errorf("expected nil PCAReducer error, got %v", err)
		}

		_, err = p.InverseTransform(ctx, [][]float32{{1, 2}})
		if err == nil || err.Error() != "PCAReducer is nil" {
			t.Errorf("expected nil PCAReducer error, got %v", err)
		}

		_, err = p.GetReconstructionError(ctx, [][]float32{{1, 2, 3}})
		if err == nil || err.Error() != "PCAReducer is nil" {
			t.Errorf("expected nil PCAReducer error, got %v", err)
		}

		// Test getter methods
		if p.OriginalDim() != 0 {
			t.Errorf("expected 0 for nil reducer OriginalDim")
		}
		if p.ReducedDim() != 0 {
			t.Errorf("expected 0 for nil reducer ReducedDim")
		}
		if p.ExplainedVarianceRatio() != nil {
			t.Errorf("expected nil for nil reducer ExplainedVarianceRatio")
		}
		if p.GetTopFeatures(0, 5) != nil {
			t.Errorf("expected nil for nil reducer GetTopFeatures")
		}
	})

	t.Run("nil embeddings", func(t *testing.T) {
		p := NewPCAReducer(&Config{TargetDim: 2})

		err := p.Fit(ctx, nil)
		if err == nil || !contains(err.Error(), "embeddings cannot be nil") {
			t.Errorf("expected nil embeddings error, got %v", err)
		}
	})

	t.Run("empty embeddings", func(t *testing.T) {
		p := NewPCAReducer(&Config{TargetDim: 2})

		err := p.Fit(ctx, [][]float32{})
		if err == nil || !contains(err.Error(), "no embeddings provided") {
			t.Errorf("expected empty embeddings error, got %v", err)
		}
	})

	t.Run("empty first embedding", func(t *testing.T) {
		p := NewPCAReducer(&Config{TargetDim: 2})

		err := p.Fit(ctx, [][]float32{{}})
		if err == nil || !contains(err.Error(), "first embedding is empty") {
			t.Errorf("expected empty first embedding error, got %v", err)
		}
	})

	t.Run("nil embedding in slice", func(t *testing.T) {
		p := NewPCAReducer(&Config{TargetDim: 2})

		embeddings := [][]float32{
			{1, 2, 3},
			nil,
			{4, 5, 6},
		}
		err := p.Fit(ctx, embeddings)
		if err == nil || !contains(err.Error(), "embedding 1 is nil") {
			t.Errorf("expected nil embedding error, got %v", err)
		}
	})

	t.Run("inconsistent dimensions", func(t *testing.T) {
		p := NewPCAReducer(&Config{TargetDim: 2})

		embeddings := [][]float32{
			{1, 2, 3},
			{4, 5},     // Wrong dimension
			{6, 7, 8},
		}
		err := p.Fit(ctx, embeddings)
		if err == nil || !contains(err.Error(), "inconsistent dimensions") {
			t.Errorf("expected inconsistent dimensions error, got %v", err)
		}
	})

	t.Run("NaN values", func(t *testing.T) {
		p := NewPCAReducer(&Config{TargetDim: 2})

		embeddings := [][]float32{
			{1, 2, 3},
			{4, float32(math.NaN()), 6},
			{7, 8, 9},
		}
		err := p.Fit(ctx, embeddings)
		if err == nil || !contains(err.Error(), "invalid value") {
			t.Errorf("expected invalid value error, got %v", err)
		}
	})

	t.Run("Inf values", func(t *testing.T) {
		p := NewPCAReducer(&Config{TargetDim: 2})

		embeddings := [][]float32{
			{1, 2, 3},
			{4, float32(math.Inf(1)), 6},
			{7, 8, 9},
		}
		err := p.Fit(ctx, embeddings)
		if err == nil || !contains(err.Error(), "invalid value") {
			t.Errorf("expected invalid value error, got %v", err)
		}
	})

	t.Run("target dimension exceeds original", func(t *testing.T) {
		p := NewPCAReducer(&Config{TargetDim: 5})

		embeddings := [][]float32{
			{1, 2, 3},
			{4, 5, 6},
			{7, 8, 9},
		}
		err := p.Fit(ctx, embeddings)
		if err == nil || !contains(err.Error(), "target dimension") {
			t.Errorf("expected target dimension error, got %v", err)
		}
	})

	t.Run("insufficient samples", func(t *testing.T) {
		p := NewPCAReducer(&Config{TargetDim: 2})

		embeddings := [][]float32{
			{1, 2, 3},
		}
		err := p.Fit(ctx, embeddings)
		if err == nil || !contains(err.Error(), "insufficient samples") {
			t.Errorf("expected insufficient samples error, got %v", err)
		}
	})

	t.Run("transform before fit", func(t *testing.T) {
		p := NewPCAReducer(&Config{TargetDim: 2})

		_, err := p.Transform(ctx, [][]float32{{1, 2, 3}})
		if err == nil || !contains(err.Error(), "not fitted yet") {
			t.Errorf("expected not fitted error, got %v", err)
		}
	})

	t.Run("transform wrong dimension", func(t *testing.T) {
		p := NewPCAReducer(&Config{TargetDim: 2})

		// Fit with 3D embeddings
		embeddings := [][]float32{
			{1, 2, 3},
			{4, 5, 6},
			{7, 8, 9},
		}
		err := p.Fit(ctx, embeddings)
		if err != nil {
			t.Fatalf("failed to fit: %v", err)
		}

		// Try to transform 4D embeddings
		_, err = p.Transform(ctx, [][]float32{{1, 2, 3, 4}})
		if err == nil || !contains(err.Error(), "wrong dimension") {
			t.Errorf("expected wrong dimension error, got %v", err)
		}
	})

	t.Run("inverse transform wrong dimension", func(t *testing.T) {
		p := NewPCAReducer(&Config{TargetDim: 2})

		// Fit with 3D embeddings
		embeddings := [][]float32{
			{1, 2, 3},
			{4, 5, 6},
			{7, 8, 9},
		}
		err := p.Fit(ctx, embeddings)
		if err != nil {
			t.Fatalf("failed to fit: %v", err)
		}

		// Try to inverse transform with wrong reduced dimension
		_, err = p.InverseTransform(ctx, [][]float32{{1, 2, 3}})
		if err == nil || !contains(err.Error(), "wrong dimension") {
			t.Errorf("expected wrong dimension error, got %v", err)
		}
	})

	t.Run("invalid component index", func(t *testing.T) {
		p := NewPCAReducer(&Config{TargetDim: 2})

		embeddings := [][]float32{
			{1, 2, 3},
			{4, 5, 6},
			{7, 8, 9},
		}
		p.Fit(ctx, embeddings)

		// Test negative index
		if p.GetTopFeatures(-1, 5) != nil {
			t.Errorf("expected nil for negative component index")
		}

		// Test out of bounds index
		if p.GetTopFeatures(10, 5) != nil {
			t.Errorf("expected nil for out of bounds component index")
		}

		// Test zero topK
		if p.GetTopFeatures(0, 0) != nil {
			t.Errorf("expected nil for zero topK")
		}
	})
}

func TestDimensionReducer_ValidationErrors(t *testing.T) {
	ctx := context.Background()

	t.Run("nil config", func(t *testing.T) {
		_, err := NewDimensionReducer(nil)
		if err == nil || !contains(err.Error(), "config cannot be nil") {
			t.Errorf("expected nil config error, got %v", err)
		}
	})

	t.Run("nil reducer methods", func(t *testing.T) {
		var dr *DimensionReducer

		// Test all methods with nil reducer
		err := dr.Learn(ctx, [][]float32{{1, 2, 3}})
		if err == nil || !contains(err.Error(), "DimensionReducer is nil") {
			t.Errorf("expected nil DimensionReducer error, got %v", err)
		}

		_, err = dr.ReduceForSearch(ctx, []float32{1, 2, 3})
		if err == nil || !contains(err.Error(), "DimensionReducer is nil") {
			t.Errorf("expected nil DimensionReducer error, got %v", err)
		}

		_, err = dr.HybridSearch(ctx, []float32{1, 2, 3}, []SearchCandidate{}, 5, testCosineSimilarity)
		if err == nil || !contains(err.Error(), "DimensionReducer is nil") {
			t.Errorf("expected nil DimensionReducer error, got %v", err)
		}

		_, err = dr.ReduceBatch(ctx, [][]float32{{1, 2, 3}})
		if err == nil || !contains(err.Error(), "DimensionReducer is nil") {
			t.Errorf("expected nil DimensionReducer error, got %v", err)
		}

		// Test getter methods
		metrics := dr.GetMetrics()
		if metrics.TotalQueries != 0 {
			t.Errorf("expected zero metrics for nil reducer")
		}

		info := dr.GetReductionInfo()
		if info.OriginalDim != 0 {
			t.Errorf("expected zero info for nil reducer")
		}

		if dr.ShouldUseReduction() {
			t.Errorf("expected false for nil reducer ShouldUseReduction")
		}

		if dr.EstimateSearchSpeedup() != 1.0 {
			t.Errorf("expected 1.0 speedup for nil reducer")
		}

		// Test UpdateHitRates with nil reducer
		dr.UpdateHitRates(0.8, 0.7) // Should not panic
	})

	t.Run("invalid search inputs", func(t *testing.T) {
		dr, err := NewDimensionReducer(&Config{TargetDim: 2})
		if err != nil {
			t.Fatalf("failed to create reducer: %v", err)
		}

		// Nil query embedding
		_, err = dr.HybridSearch(ctx, nil, []SearchCandidate{}, 5, testCosineSimilarity)
		if err == nil || !contains(err.Error(), "query embedding cannot be nil") {
			t.Errorf("expected nil query error, got %v", err)
		}

		// Empty query embedding
		_, err = dr.HybridSearch(ctx, []float32{}, []SearchCandidate{}, 5, testCosineSimilarity)
		if err == nil || !contains(err.Error(), "query embedding cannot be empty") {
			t.Errorf("expected empty query error, got %v", err)
		}

		// Invalid query values
		_, err = dr.HybridSearch(ctx, []float32{1, float32(math.NaN()), 3}, []SearchCandidate{}, 5, testCosineSimilarity)
		if err == nil || !contains(err.Error(), "invalid value") {
			t.Errorf("expected invalid value error, got %v", err)
		}

		// Nil candidates
		_, err = dr.HybridSearch(ctx, []float32{1, 2, 3}, nil, 5, testCosineSimilarity)
		if err == nil || !contains(err.Error(), "candidates cannot be nil") {
			t.Errorf("expected nil candidates error, got %v", err)
		}

		// Invalid topK
		_, err = dr.HybridSearch(ctx, []float32{1, 2, 3}, []SearchCandidate{}, 0, testCosineSimilarity)
		if err == nil || !contains(err.Error(), "topK must be positive") {
			t.Errorf("expected invalid topK error, got %v", err)
		}

		// Nil similarity function
		_, err = dr.HybridSearch(ctx, []float32{1, 2, 3}, []SearchCandidate{}, 5, nil)
		if err == nil || !contains(err.Error(), "similarity function cannot be nil") {
			t.Errorf("expected nil similarity function error, got %v", err)
		}
	})

	t.Run("reduce before learn", func(t *testing.T) {
		dr, err := NewDimensionReducer(&Config{TargetDim: 2})
		if err != nil {
			t.Fatalf("failed to create reducer: %v", err)
		}

		_, err = dr.ReduceForSearch(ctx, []float32{1, 2, 3})
		if err == nil || !contains(err.Error(), "reducer not learned yet") {
			t.Errorf("expected not learned error, got %v", err)
		}
	})

	t.Run("invalid hit rates", func(t *testing.T) {
		dr, err := NewDimensionReducer(&Config{TargetDim: 2})
		if err != nil {
			t.Fatalf("failed to create reducer: %v", err)
		}

		// Should not update with invalid rates
		dr.UpdateHitRates(-0.5, 0.7)  // Negative rate
		dr.UpdateHitRates(1.5, 0.7)   // Rate > 1
		dr.UpdateHitRates(0.8, -0.2)  // Negative rate
		dr.UpdateHitRates(0.8, 1.2)   // Rate > 1

		metrics := dr.GetMetrics()
		if metrics.HitRateBeforeReduction != 0 || metrics.HitRateAfterReduction != 0 {
			t.Errorf("expected zero hit rates after invalid updates")
		}
	})

	t.Run("dimension mismatch in search", func(t *testing.T) {
		dr, err := NewDimensionReducer(&Config{TargetDim: 2})
		if err != nil {
			t.Fatalf("failed to create reducer: %v", err)
		}

		// Learn with 3D embeddings
		embeddings := [][]float32{
			{1, 2, 3},
			{4, 5, 6},
			{7, 8, 9},
		}
		err = dr.Learn(ctx, embeddings)
		if err != nil {
			t.Fatalf("failed to learn: %v", err)
		}

		// Try to reduce 4D embedding
		_, err = dr.ReduceForSearch(ctx, []float32{1, 2, 3, 4})
		if err == nil || !contains(err.Error(), "dimension mismatch") {
			t.Errorf("expected dimension mismatch error, got %v", err)
		}
	})
}

// Helper function for testing
func contains(s, substr string) bool {
	return len(s) >= len(substr) && s[:len(substr)] == substr || 
		   len(s) >= len(substr) && contains(s[1:], substr)
}

// testCosineSimilarity is a test similarity function
func testCosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	
	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}
	
	if normA == 0 || normB == 0 {
		return 0
	}
	
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}