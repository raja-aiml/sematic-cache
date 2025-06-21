// Package main demonstrates dimension reduction with hybrid search and A/B testing
package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/raja-aiml/sematic-cache/core"
	"github.com/raja-aiml/sematic-cache/core/reduction"
	"github.com/raja-aiml/sematic-cache/openai"
)

func main() {
	// Initialize OpenAI client for embeddings
	apiKey := "your-openai-api-key" // Set via environment variable in production
	client := openai.NewClient(apiKey)

	// Create embedding function
	embedFunc := func(text string) ([]float32, error) {
		return client.Embedding(context.Background(), text)
	}

	// Step 1: Create dimension reduction configuration using factory pattern
	reductionConfig := reduction.DimensionReducerConfig{
		ReducerConfig: reduction.ReducerConfig{
			OutputDimensions: 384,  // Reduce from 1536 to 384 dimensions
			VarianceRetained: 0.95, // Retain 95% variance
		},
		Type:               reduction.PCAReducerType, // Use PCA algorithm
		EnableOptimization: true,                     // Use optimized Gonum implementation
	}

	// Create dimension reducer using factory
	reducer, err := reduction.NewDimensionReducerWithFactory(reductionConfig)
	if err != nil {
		log.Fatalf("Failed to create dimension reducer: %v", err)
	}

	// Step 2: Set up A/B testing
	defaultStrategy := reduction.Strategy{
		ID:        "baseline",
		Name:      "No Reduction",
		TargetDim: 1536,
		Algorithm: "none",
		UseHybrid: false,
	}

	abTestManager := reduction.NewABTestManager(defaultStrategy)

	// Define test strategies
	strategies := []reduction.Strategy{
		defaultStrategy,
		{
			ID:        "pca_384",
			Name:      "PCA 384 Dimensions",
			TargetDim: 384,
			Algorithm: string(reduction.PCAReducerType),
			UseHybrid: true,
		},
		{
			ID:        "pca_gonum_256",
			Name:      "PCA Gonum 256 Dimensions",
			TargetDim: 256,
			Algorithm: string(reduction.PCAGonumReducerType),
			UseHybrid: true,
		},
		{
			ID:        "incremental_pca_384",
			Name:      "Incremental PCA 384 Dimensions",
			TargetDim: 384,
			Algorithm: string(reduction.IncrementalPCAReducerType),
			UseHybrid: true,
		},
	}

	// Configure A/B test
	testConfig := reduction.ABTestConfig{
		MinImpressions:        1000,
		MinDurationHours:      1.0,
		ConfidenceLevel:       0.95,
		SignificanceThreshold: 0.95,
		MetricWeights: reduction.MetricWeights{
			HitRate: 0.4,
			Latency: 0.3,
			Memory:  0.2,
			Quality: 0.1,
		},
	}

	// Create and start A/B test
	test, err := abTestManager.CreateTest(testConfig, strategies, []float64{0.33, 0.33, 0.34})
	if err != nil {
		log.Fatalf("Failed to create A/B test: %v", err)
	}

	if err := abTestManager.StartTest(test.ID); err != nil {
		log.Fatalf("Failed to start A/B test: %v", err)
	}

	// Step 3: Create cache with dimension reduction and A/B testing
	cache, err := core.NewCache(10000,
		core.WithMinSimilarity(0.8),
		core.WithEmbeddingFunc(embedFunc),
		core.WithDimensionReduction(reducer),
		core.WithABTestManager(abTestManager),
	)
	if err != nil {
		log.Fatalf("failed to create cache: %v", err)
	}

	// Step 4: Populate cache with sample data
	fmt.Println("Populating cache with sample data...")
	sampleQueries := []struct {
		prompt string
		answer string
	}{
		{"What is machine learning?", "Machine learning is a subset of AI that enables systems to learn from data."},
		{"Explain neural networks", "Neural networks are computing systems inspired by biological neural networks."},
		{"What is deep learning?", "Deep learning is a subset of ML using multi-layered neural networks."},
		{"How does AI work?", "AI works by processing data through algorithms to make predictions."},
		{"What is natural language processing?", "NLP is AI that helps computers understand human language."},
		// Add more sample data...
	}

	for _, sample := range sampleQueries {
		if err := cache.SetPrompt(sample.prompt, sample.answer); err != nil {
			log.Printf("Failed to cache %s: %v", sample.prompt, err)
		}
	}

	// Step 5: Train dimension reducer on cached embeddings
	fmt.Println("Training dimension reducer...")
	ctx := context.Background()
	if err := cache.TrainDimensionReducer(ctx); err != nil {
		log.Fatalf("Failed to train dimension reducer: %v", err)
	}

	// Get reduction info
	reductionInfo := reducer.GetReductionInfo()
	fmt.Printf("Dimension Reduction Info:\n")
	fmt.Printf("  Original Dimensions: %d\n", reductionInfo.OriginalDim)
	fmt.Printf("  Reduced Dimensions: %d\n", reductionInfo.ReducedDim)
	fmt.Printf("  Variance Explained: %.2f%%\n", reductionInfo.VarianceExplained*100)
	fmt.Printf("  Compression Ratio: %.2f\n", reductionInfo.CompressionRatio)

	// Step 6: Test hybrid search with different queries
	testQueries := []string{
		"What is ML?",
		"Explain deep neural networks",
		"How do computers understand language?",
		"Tell me about artificial intelligence",
	}

	fmt.Println("\nTesting hybrid search...")
	for _, query := range testQueries {
		// Get embedding for query
		embedding, err := embedFunc(query)
		if err != nil {
			log.Printf("Failed to get embedding for %s: %v", query, err)
			continue
		}

		// Search for similar entries
		results := cache.GetTopKByEmbedding(embedding, 3)

		fmt.Printf("\nQuery: %s\n", query)
		fmt.Println("Results:")
		for i, result := range results {
			fmt.Printf("  %d. %.3f - %s: %s...\n",
				i+1,
				result.Similarity,
				result.Prompt,
				result.Answer[:min(50, len(result.Answer))])
		}
	}

	// Step 7: Monitor A/B test results
	fmt.Println("\n\nMonitoring A/B Test Results...")
	dashboard := reduction.NewMonitoringDashboard(abTestManager)

	// Simulate some queries to generate metrics
	for i := 0; i < 100; i++ {
		query := testQueries[rand.Intn(len(testQueries))]
		embedding, _ := embedFunc(query)
		_ = cache.GetTopKByEmbedding(embedding, 1)
		time.Sleep(10 * time.Millisecond)
	}

	// Get live metrics
	liveMetrics := dashboard.GetLiveMetrics()
	if liveMetrics != nil {
		fmt.Printf("\nLive A/B Test Metrics:\n")
		fmt.Printf("Test ID: %s\n", liveMetrics.TestID)
		fmt.Printf("Duration: %v\n", liveMetrics.Duration)

		for _, stratMetrics := range liveMetrics.Strategies {
			fmt.Printf("\nStrategy: %s\n", stratMetrics.StrategyName)
			fmt.Printf("  Impressions: %d\n", stratMetrics.Impressions)
			fmt.Printf("  Hit Rate: %.2f%%\n", stratMetrics.HitRate*100)
			fmt.Printf("  Error Rate: %.2f%%\n", stratMetrics.ErrorRate*100)
		}
	}

	// Check if test is complete
	complete, summary := abTestManager.CheckTestCompletion(test.ID)
	if complete && summary != nil {
		fmt.Printf("\nA/B Test Complete!\n")
		fmt.Printf("Winning Strategy: %s\n", summary.WinningStrategy.Name)
		fmt.Printf("Statistical Significance: %.2f\n", summary.StatisticalSignificance)

		fmt.Printf("\nDetailed Results:\n")
		for _, result := range summary.Results {
			fmt.Printf("\nStrategy: %s\n", result.Strategy.Name)
			fmt.Printf("  Hit Rate: %.2f%%\n", result.HitRate*100)
			fmt.Printf("  Avg Latency: %.2fms\n", result.AvgLatencyMs)
			fmt.Printf("  Avg Similarity: %.3f\n", result.AvgSimilarityScore)
			fmt.Printf("  Score: %.3f\n", result.Score)
		}
	}

	// Step 8: Get quality metrics
	metrics := cache.GetDimensionReductionMetrics()
	if metrics != nil {
		fmt.Printf("\n\nDimension Reduction Quality Metrics:\n")
		fmt.Printf("  Metrics collected successfully\n")
		// Note: Specific metrics are unexported, but the reducer info is available
		info := reducer.GetReductionInfo()
		fmt.Printf("  Variance Explained: %.2f%%\n", info.VarianceExplained*100)
		fmt.Printf("  Compression Ratio: %.2f\n", info.CompressionRatio)
	}

	// Estimate speedup
	speedup := reducer.EstimateSearchSpeedup()
	fmt.Printf("\nEstimated Search Speedup: %.2fx\n", speedup)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
