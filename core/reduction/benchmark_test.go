package reduction

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
)

// generateRandomEmbeddings creates random embeddings for testing
func generateRandomEmbeddings(count, dim int) [][]float32 {
	embeddings := make([][]float32, count)
	for i := 0; i < count; i++ {
		embeddings[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			embeddings[i][j] = rand.Float32()*2 - 1 // [-1, 1]
		}
	}
	return embeddings
}

// generateCandidates creates search candidates for testing
func generateCandidates(embeddings [][]float32, reducedEmbeddings [][]float32) []SearchCandidate {
	candidates := make([]SearchCandidate, len(embeddings))
	for i := 0; i < len(embeddings); i++ {
		candidates[i] = SearchCandidate{
			ID:               fmt.Sprintf("doc_%d", i),
			Embedding:        embeddings[i],
			ReducedEmbedding: nil,
			Metadata:         map[string]interface{}{"index": i},
		}
		if i < len(reducedEmbeddings) {
			candidates[i].ReducedEmbedding = reducedEmbeddings[i]
		}
	}
	return candidates
}

// BenchmarkPCAImplementations compares naive vs Gonum PCA
func BenchmarkPCAImplementations(b *testing.B) {
	sizes := []struct {
		name       string
		numSamples int
		dim        int
		targetDim  int
	}{
		{"Small_100x384x64", 100, 384, 64},
		{"Medium_1000x768x128", 1000, 768, 128},
		{"Large_5000x1536x384", 5000, 1536, 384},
	}

	for _, size := range sizes {
		embeddings := generateRandomEmbeddings(size.numSamples, size.dim)
		config := &Config{
			TargetDim:         size.targetDim,
			VarianceThreshold: 0,
			Standardize:       false,
		}

		b.Run(size.name+"_Naive", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				pca := NewPCAReducer(config)
				ctx := context.Background()
				err := pca.Fit(ctx, embeddings)
				if err != nil {
					b.Fatal(err)
				}
			}
		})

		b.Run(size.name+"_Gonum", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				pca := NewPCAGonumReducer(config)
				ctx := context.Background()
				err := pca.Fit(ctx, embeddings)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkTopKSelection compares sorting vs heap-based selection
func BenchmarkTopKSelection(b *testing.B) {
	sizes := []struct {
		name       string
		candidates int
		topK       int
	}{
		{"Small_1000_10", 1000, 10},
		{"Medium_10000_50", 10000, 50},
		{"Large_100000_100", 100000, 100},
	}

	for _, size := range sizes {
		// Generate scored candidates
		candidates := make([]scoredCandidate, size.candidates)
		for i := 0; i < size.candidates; i++ {
			candidates[i] = scoredCandidate{
				candidate: SearchCandidate{
					ID: fmt.Sprintf("doc_%d", i),
				},
				similarity: rand.Float64(),
			}
		}

		b.Run(size.name+"_Sorting", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Copy candidates to avoid modifying original
				candCopy := make([]scoredCandidate, len(candidates))
				copy(candCopy, candidates)
				
				// Traditional O(n log n) sorting
				sortBySimilarity(candCopy)
				
				// Take top-K
				if size.topK < len(candCopy) {
					candCopy = candCopy[:size.topK]
				}
			}
		})

		b.Run(size.name+"_Heap", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Heap-based O(n log k) selection
				selector := NewTopKSelector(size.topK)
				_ = selector.SelectTopK(candidates)
			}
		})
	}
}

// BenchmarkHybridSearch compares regular vs optimized hybrid search
func BenchmarkHybridSearch(b *testing.B) {
	// Setup
	ctx := context.Background()
	config := &Config{
		TargetDim:         384,
		VarianceThreshold: 0.95,
	}

	// Generate training data
	trainEmbeddings := generateRandomEmbeddings(5000, 1536)

	// Train regular reducer
	regularReducer, _ := NewDimensionReducer(config)
	regularReducer.Learn(ctx, trainEmbeddings)

	// Train optimized reducer
	optimizedReducer, _ := NewOptimizedDimensionReducer(config)
	optimizedReducer.Learn(ctx, trainEmbeddings)

	// Generate test data
	testSizes := []struct {
		name       string
		candidates int
		topK       int
	}{
		{"Small_1000_10", 1000, 10},
		{"Medium_10000_50", 10000, 50},
		{"Large_50000_100", 50000, 100},
	}

	for _, size := range testSizes {
		// Generate candidates
		fullEmbeddings := generateRandomEmbeddings(size.candidates, 1536)
		reducedEmbeddings, _ := regularReducer.ReduceBatch(ctx, fullEmbeddings)
		candidates := generateCandidates(fullEmbeddings, reducedEmbeddings)

		// Generate query
		queryEmbedding := generateRandomEmbeddings(1, 1536)[0]

		b.Run(size.name+"_Regular", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := regularReducer.HybridSearch(ctx, queryEmbedding, candidates, size.topK, CosineSimilarity)
				if err != nil {
					b.Fatal(err)
				}
			}
		})

		b.Run(size.name+"_Optimized", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := optimizedReducer.OptimizedHybridSearch(ctx, queryEmbedding, candidates, size.topK, CosineSimilarity)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkObjectPooling compares performance with and without object pooling
func BenchmarkObjectPooling(b *testing.B) {
	ctx := context.Background()
	config := &Config{
		TargetDim: 384,
	}

	// Setup optimized reducer
	reducer, _ := NewOptimizedDimensionReducer(config)
	
	// Train on sample data
	trainEmbeddings := generateRandomEmbeddings(1000, 1536)
	reducer.Learn(ctx, trainEmbeddings)

	// Generate test data
	fullEmbeddings := generateRandomEmbeddings(10000, 1536)
	reducedEmbeddings, _ := reducer.ReduceBatch(ctx, fullEmbeddings)
	candidates := generateCandidates(fullEmbeddings, reducedEmbeddings)
	queryEmbedding := generateRandomEmbeddings(1, 1536)[0]

	b.Run("WithPooling", func(b *testing.B) {
		reducer.EnableObjectPooling(true)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := reducer.OptimizedHybridSearch(ctx, queryEmbedding, candidates, 50, CosineSimilarity)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("WithoutPooling", func(b *testing.B) {
		reducer.EnableObjectPooling(false)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := reducer.OptimizedHybridSearch(ctx, queryEmbedding, candidates, 50, CosineSimilarity)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkBatchProcessing compares single vs batch processing
func BenchmarkBatchProcessing(b *testing.B) {
	ctx := context.Background()
	config := &Config{
		TargetDim: 384,
	}

	pca := NewPCAGonumReducer(config)
	
	// Train on sample data
	trainEmbeddings := generateRandomEmbeddings(1000, 1536)
	pca.Fit(ctx, trainEmbeddings)

	// Test different batch sizes
	batchSizes := []int{1, 10, 100, 1000}
	
	for _, batchSize := range batchSizes {
		testEmbeddings := generateRandomEmbeddings(batchSize, 1536)
		
		b.Run(fmt.Sprintf("BatchSize_%d", batchSize), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := pca.BatchTransform(ctx, testEmbeddings, 100)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkMemoryUsage measures memory allocation improvements
func BenchmarkMemoryUsage(b *testing.B) {
	ctx := context.Background()
	config := &Config{
		TargetDim: 384,
	}

	// Generate test data
	candidates := generateCandidates(
		generateRandomEmbeddings(10000, 1536),
		generateRandomEmbeddings(10000, 384),
	)
	queryEmbedding := generateRandomEmbeddings(1, 1536)[0]

	b.Run("Regular", func(b *testing.B) {
		reducer, _ := NewDimensionReducer(config)
		trainData := generateRandomEmbeddings(1000, 1536)
		reducer.Learn(ctx, trainData)
		
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _ = reducer.HybridSearch(ctx, queryEmbedding, candidates, 50, CosineSimilarity)
		}
	})

	b.Run("Optimized", func(b *testing.B) {
		reducer, _ := NewOptimizedDimensionReducer(config)
		trainData := generateRandomEmbeddings(1000, 1536)
		reducer.Learn(ctx, trainData)
		
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _ = reducer.OptimizedHybridSearch(ctx, queryEmbedding, candidates, 50, CosineSimilarity)
		}
	})
}

// CosineSimilarity computes cosine similarity between two vectors
func CosineSimilarity(a, b []float32) float64 {
	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	
	if normA == 0 || normB == 0 {
		return 0
	}
	
	return dotProduct / (Sqrt(normA) * Sqrt(normB))
}

// Sqrt is a helper function for square root
func Sqrt(x float64) float64 {
	if x < 0 {
		return 0
	}
	z := x
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}