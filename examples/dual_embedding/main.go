// Package main demonstrates dual embedding storage for optimal performance
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/raja-aiml/sematic-cache/core"
	"github.com/raja-aiml/sematic-cache/core/reduction"
	"github.com/raja-aiml/sematic-cache/openai"
)

func main() {
	// Initialize OpenAI client
	apiKey := "your-openai-api-key"
	client := openai.NewClient(apiKey)

	embedFunc := func(text string) ([]float32, error) {
		return client.Embedding(context.Background(), text)
	}

	// Step 1: Create dimension reducer using factory pattern
	reductionConfig := reduction.DimensionReducerConfig{
		ReducerConfig: reduction.ReducerConfig{
			OutputDimensions: 384,  // 75% reduction  
			VarianceRetained: 0.95, // Retain 95% variance
		},
		Type:               reduction.PCAReducerType,  // Use standard PCA
		EnableOptimization: false,                     // Use standard implementation for demo
	}

	reducer, err := reduction.NewDimensionReducerWithFactory(reductionConfig)
	if err != nil {
		log.Fatalf("Failed to create reducer: %v", err)
	}

	// Step 2: Create cache with dimension reduction
	// This automatically enables dual embedding storage
	cache := core.NewCache(1000,
		core.WithEmbeddingFunc(embedFunc),
		core.WithDimensionReduction(reducer),
		core.WithMinSimilarity(0.8),
	)

	ctx := context.Background()

	// Step 3: Add initial entries (only full embeddings stored initially)
	fmt.Println("Adding initial entries...")
	entries := []struct {
		prompt string
		answer string
	}{
		{"What is machine learning?", "ML is a subset of AI that learns from data."},
		{"Explain neural networks", "Neural networks are computing systems inspired by biological neurons."},
		{"What is deep learning?", "Deep learning uses multi-layered neural networks."},
		{"How does AI work?", "AI processes data through algorithms to make decisions."},
		{"What is NLP?", "Natural Language Processing helps computers understand human language."},
		{"Explain computer vision", "Computer vision enables machines to interpret visual information."},
		{"What is reinforcement learning?", "RL is learning through trial and error with rewards."},
		{"How do transformers work?", "Transformers use attention mechanisms for sequence processing."},
		{"What is gradient descent?", "Gradient descent optimizes model parameters by minimizing loss."},
		{"Explain backpropagation", "Backpropagation calculates gradients for neural network training."},
	}

	for _, e := range entries {
		if err := cache.SetPrompt(e.prompt, e.answer); err != nil {
			log.Printf("Failed to cache %s: %v", e.prompt, err)
		}
	}

	// Check initial stats
	stats := cache.GetDualEmbeddingStats()
	fmt.Printf("\nInitial Dual Embedding Stats:\n")
	fmt.Printf("  Total entries: %d\n", stats.TotalEntries)
	fmt.Printf("  With full embedding: %d\n", stats.WithFullEmbedding)
	fmt.Printf("  With reduced embedding: %d\n", stats.WithReducedEmbedding)
	fmt.Printf("  With both embeddings: %d (%.1f%%)\n", stats.WithBothEmbeddings, stats.CoveragePercent)

	// Step 4: Train dimension reducer
	fmt.Println("\nTraining dimension reducer...")
	if err := cache.TrainDimensionReducer(ctx); err != nil {
		log.Fatalf("Failed to train reducer: %v", err)
	}

	// Check stats after training
	stats = cache.GetDualEmbeddingStats()
	fmt.Printf("\nPost-Training Dual Embedding Stats:\n")
	fmt.Printf("  Total entries: %d\n", stats.TotalEntries)
	fmt.Printf("  With full embedding: %d\n", stats.WithFullEmbedding)
	fmt.Printf("  With reduced embedding: %d\n", stats.WithReducedEmbedding)
	fmt.Printf("  With both embeddings: %d (%.1f%%)\n", stats.WithBothEmbeddings, stats.CoveragePercent)

	// Step 5: Add new entries (both embeddings stored automatically)
	fmt.Println("\nAdding new entries after training...")
	newEntries := []struct {
		prompt string
		answer string
	}{
		{"What are CNNs?", "Convolutional Neural Networks are specialized for image processing."},
		{"Explain RNNs", "Recurrent Neural Networks process sequential data."},
	}

	for _, e := range newEntries {
		if err := cache.SetPrompt(e.prompt, e.answer); err != nil {
			log.Printf("Failed to cache %s: %v", e.prompt, err)
		}
	}

	// Final stats
	stats = cache.GetDualEmbeddingStats()
	fmt.Printf("\nFinal Dual Embedding Stats:\n")
	fmt.Printf("  Total entries: %d\n", stats.TotalEntries)
	fmt.Printf("  With full embedding: %d\n", stats.WithFullEmbedding)
	fmt.Printf("  With reduced embedding: %d\n", stats.WithReducedEmbedding)
	fmt.Printf("  With both embeddings: %d (%.1f%%)\n", stats.WithBothEmbeddings, stats.CoveragePercent)

	// Step 6: Demonstrate hybrid search performance
	fmt.Println("\n\nDemonstrating Hybrid Search Performance:")

	testQueries := []string{
		"What are Convolutional Neural Networks?",
		"How do transformers process sequences?",
		"Explain machine learning",
	}

	for _, query := range testQueries {
		fmt.Printf("\nQuery: %s\n", query)

		// Get embedding
		embedding, err := embedFunc(query)
		if err != nil {
			log.Printf("Failed to get embedding: %v", err)
			continue
		}

		// Time the search
		start := time.Now()
		results := cache.GetTopKByEmbedding(embedding, 3)
		searchTime := time.Since(start)

		fmt.Printf("Search time: %v\n", searchTime)
		fmt.Println("Results:")
		for i, result := range results {
			fmt.Printf("  %d. [%.3f] %s\n", i+1, result.Similarity, result.Prompt)
		}
	}

	// Step 7: Ensure all entries have both embeddings
	fmt.Println("\n\nEnsuring all entries have both embeddings...")
	if err := cache.EnsureReducedEmbeddings(ctx); err != nil {
		log.Printf("Warning: %v", err)
	}

	// Verify complete dual embedding coverage
	stats = cache.GetDualEmbeddingStats()
	fmt.Printf("\nVerified Dual Embedding Coverage:\n")
	fmt.Printf("  Coverage: %.1f%% (%d/%d entries have both embeddings)\n",
		stats.CoveragePercent, stats.WithBothEmbeddings, stats.TotalEntries)

	// Step 8: Show memory savings
	info := reducer.GetReductionInfo()
	originalMemory := stats.TotalEntries * info.OriginalDim * 4 / 1024 / 1024 // MB
	reducedMemory := stats.TotalEntries * info.ReducedDim * 4 / 1024 / 1024   // MB
	totalMemory := originalMemory + reducedMemory

	fmt.Printf("\nMemory Usage:\n")
	fmt.Printf("  Original embeddings only: %.2f MB\n", float64(originalMemory))
	fmt.Printf("  Reduced embeddings only: %.2f MB\n", float64(reducedMemory))
	fmt.Printf("  Both embeddings: %.2f MB\n", float64(totalMemory))
	fmt.Printf("  Memory overhead for dual storage: %.1f%%\n",
		float64(reducedMemory)/float64(originalMemory)*100)

	// The overhead is worth it for:
	// 1. 3-4x faster initial search
	// 2. High accuracy through re-ranking
	// 3. Flexibility to disable reduction if needed
	// 4. A/B testing capabilities
}
