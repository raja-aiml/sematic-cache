// Package main demonstrates using the reducer factory pattern to select different algorithms
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

	// Sample data
	entries := []struct {
		prompt string
		answer string
	}{
		{"What is machine learning?", "ML is a subset of AI that learns from data."},
		{"Explain neural networks", "Neural networks are computing systems inspired by biological neurons."},
		{"What is deep learning?", "Deep learning uses multi-layered neural networks."},
		{"How does AI work?", "AI processes data through algorithms to make decisions."},
		{"What is NLP?", "Natural Language Processing helps computers understand human language."},
	}

	// Demonstrate different reducer types
	reducerTypes := []struct {
		name        string
		reducerType reduction.ReducerType
		config      reduction.DimensionReducerConfig
	}{
		{
			name:        "Standard PCA",
			reducerType: reduction.PCAReducerType,
			config: reduction.DimensionReducerConfig{
				ReducerConfig: reduction.ReducerConfig{
					OutputDimensions: 384,
					VarianceRetained: 0.95,
				},
				Type:               reduction.PCAReducerType,
				EnableOptimization: false,
			},
		},
		{
			name:        "Optimized PCA (Gonum)",
			reducerType: reduction.PCAGonumReducerType,
			config: reduction.DimensionReducerConfig{
				ReducerConfig: reduction.ReducerConfig{
					OutputDimensions: 384,
					VarianceRetained: 0.95,
				},
				Type:               reduction.PCAReducerType,
				EnableOptimization: true, // This will use Gonum implementation
			},
		},
		{
			name:        "Incremental PCA",
			reducerType: reduction.IncrementalPCAReducerType,
			config: reduction.DimensionReducerConfig{
				ReducerConfig: reduction.ReducerConfig{
					OutputDimensions: 384,
					VarianceRetained: 0.95,
				},
				Type:               reduction.IncrementalPCAReducerType,
				EnableOptimization: false,
			},
		},
	}

	ctx := context.Background()

	// Test each reducer type
	for _, rt := range reducerTypes {
		fmt.Printf("\n=== Testing %s ===\n", rt.name)

		// Create reducer using factory
		reducer, err := reduction.NewDimensionReducerWithFactory(rt.config)
		if err != nil {
			log.Printf("Failed to create %s reducer: %v", rt.name, err)
			continue
		}

		// Create cache with the reducer
		cache, err := core.NewCache(100,
			core.WithEmbeddingFunc(embedFunc),
			core.WithDimensionReduction(reducer),
			core.WithMinSimilarity(0.8),
		)
		if err != nil {
			log.Printf("failed to create cache: %v", err)
			continue
		}

		// Populate cache
		fmt.Printf("Populating cache...\n")
		for _, e := range entries {
			if err := cache.SetPrompt(e.prompt, e.answer); err != nil {
				log.Printf("Failed to cache %s: %v", e.prompt, err)
			}
		}

		// Train reducer
		fmt.Printf("Training %s reducer...\n", rt.name)
		start := time.Now()
		if err := cache.TrainDimensionReducer(ctx); err != nil {
			log.Printf("Failed to train reducer: %v", err)
			continue
		}
		trainTime := time.Since(start)

		// Get reduction info
		info := reducer.GetReductionInfo()
		fmt.Printf("Training completed in %v\n", trainTime)
		fmt.Printf("Reduction info:\n")
		fmt.Printf("  Original dimensions: %d\n", info.OriginalDim)
		fmt.Printf("  Reduced dimensions: %d\n", info.ReducedDim)
		fmt.Printf("  Variance explained: %.2f%%\n", info.VarianceExplained*100)
		fmt.Printf("  Compression ratio: %.2f\n", info.CompressionRatio)

		// Test search performance
		testQuery := "How do neural networks learn?"
		embedding, err := embedFunc(testQuery)
		if err != nil {
			log.Printf("Failed to get embedding: %v", err)
			continue
		}

		// Time the search
		start = time.Now()
		results := cache.GetTopKByEmbedding(embedding, 3)
		searchTime := time.Since(start)

		fmt.Printf("\nSearch results for: \"%s\"\n", testQuery)
		fmt.Printf("Search time: %v\n", searchTime)
		for i, result := range results {
			fmt.Printf("  %d. [%.3f] %s\n", i+1, result.Similarity, result.Prompt)
		}
	}

	// Demonstrate creating a custom reducer configuration
	fmt.Printf("\n\n=== Custom Reducer Configuration ===\n")

	// Create factory
	factory := reduction.NewReducerFactory()

	// Show available reducers
	fmt.Printf("Available reducer types:\n")
	for _, rt := range factory.GetAvailableReducers() {
		fmt.Printf("  - %s\n", rt)
	}

	// Create a custom configuration
	customConfig := reduction.ReducerConfig{
		OutputDimensions: 256,  // More aggressive reduction
		VarianceRetained: 0.90, // Lower variance threshold
	}

	// Create reducer directly using factory
	customReducer, err := factory.CreateReducer(reduction.PCAGonumReducerType, customConfig)
	if err != nil {
		log.Fatalf("Failed to create custom reducer: %v", err)
	}

	fmt.Printf("\nCreated custom reducer with:\n")
	fmt.Printf("  Type: %s\n", reduction.PCAGonumReducerType)
	fmt.Printf("  Output dimensions: %d\n", customConfig.OutputDimensions)
	fmt.Printf("  Variance retained: %.0f%%\n", customConfig.VarianceRetained*100)

	// The custom reducer implements the Reducer interface
	// In production, wrap it with NewDimensionReducerWithFactory for full functionality
	_ = customReducer // Avoid unused variable warning

	fmt.Printf("\nFactory pattern advantages:\n")
	fmt.Printf("  1. Easy algorithm selection\n")
	fmt.Printf("  2. Consistent interface across implementations\n")
	fmt.Printf("  3. Simple to add new algorithms\n")
	fmt.Printf("  4. Decoupled from specific implementations\n")
}
