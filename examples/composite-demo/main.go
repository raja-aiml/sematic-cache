// Package main demonstrates the three-tier composite backend
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/raja-aiml/sematic-cache/config"
	"github.com/raja-aiml/sematic-cache/core"
	"github.com/raja-aiml/sematic-cache/openai"
	"github.com/raja-aiml/sematic-cache/storage"
)

func main() {
	configFile := flag.String("config", "../../config/composite-example.yml", "Configuration file path")
	flag.Parse()

	// Load configuration
	cfg, err := config.LoadConfig(*configFile)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Create OpenAI client for embeddings
	openaiClient := openai.NewClient(cfg.OpenAI.APIKey)
	embedFunc := func(prompt string) ([]float32, error) {
		return openaiClient.Embedding(context.Background(), prompt)
	}

	// Create composite backend
	cache, err := storage.NewBackend(cfg, embedFunc)
	if err != nil {
		log.Fatalf("Failed to create cache backend: %v", err)
	}

	// Example 1: Basic caching across tiers
	fmt.Println("=== Example 1: Basic Caching ===")
	queries := []string{
		"What is machine learning?",
		"Explain artificial intelligence",
		"How does deep learning work?",
	}

	// Simulate API responses
	for _, query := range queries {
		answer := fmt.Sprintf("This is the answer to: %s", query)
		cache.SetPromptWithModel(query, answer, "gpt-3.5-turbo", "v1")
		fmt.Printf("Cached: %s\n", query)
	}

	// Example 2: Demonstrate cache hits from different tiers
	fmt.Println("\n=== Example 2: Cache Hits ===")
	
	// First access - should hit memory (L1)
	start := time.Now()
	answer, found := cache.Get("What is machine learning?")
	elapsed := time.Since(start)
	if found {
		fmt.Printf("L1 Hit (%.2fμs): %s\n", float64(elapsed.Nanoseconds())/1000, answer[:30]+"...")
	}

	// Simulate memory eviction by flushing just the memory tier
	// In real scenario, this would happen naturally with LRU eviction
	
	// Example 3: Similarity search with caching
	fmt.Println("\n=== Example 3: Similarity Search ===")
	
	// Query with similar but not exact prompt
	similarQuery := "What is ML?"
	start = time.Now()
	answer, found = cache.Get(similarQuery)
	elapsed = time.Since(start)
	if found {
		fmt.Printf("Similarity match (%.2fms): Found answer for '%s'\n", 
			float64(elapsed.Nanoseconds())/1000000, similarQuery)
		fmt.Printf("Answer: %s\n", answer[:30]+"...")
	}

	// The similar query is now cached with exact match
	start = time.Now()
	answer, found = cache.Get(similarQuery)
	elapsed = time.Since(start)
	if found {
		fmt.Printf("Exact match after caching (%.2fμs): %s\n", 
			float64(elapsed.Nanoseconds())/1000, answer[:30]+"...")
	}

	// Example 4: View tier statistics
	fmt.Println("\n=== Example 4: Tier Statistics ===")
	if composite, ok := cache.(*storage.CompositeBackend); ok {
		for _, tier := range composite.GetTiers() {
			fmt.Printf("Tier %s:\n", tier.Name)
			fmt.Printf("  Type: %s\n", tier.Type)
			fmt.Printf("  Hits: %d, Misses: %d\n", tier.Hits, tier.Misses)
			fmt.Printf("  Hit Rate: %.2f%%\n", tier.HitRate*100)
			fmt.Printf("  Vector Search: %v\n", tier.Capabilities.SupportsVectorSearch)
			fmt.Printf("  Avg Latency: %s\n", time.Duration(tier.Capabilities.AverageLatencyNs))
			fmt.Println()
		}
	}

	// Example 5: Bulk similarity search
	fmt.Println("=== Example 5: Top-K Search ===")
	embedding, _ := embedFunc("machine learning and AI")
	results := cache.GetTopKByEmbedding(embedding, 3)
	
	fmt.Printf("Found %d similar items:\n", len(results))
	for i, result := range results {
		fmt.Printf("%d. %s (similarity: %.3f)\n", i+1, result.Prompt, result.Similarity)
	}

	// Overall statistics
	hits, misses, hitRate := cache.Stats()
	fmt.Printf("\n=== Overall Statistics ===\n")
	fmt.Printf("Total Hits: %d\n", hits)
	fmt.Printf("Total Misses: %d\n", misses)
	fmt.Printf("Hit Rate: %.2f%%\n", hitRate*100)
}

// Helper function to simulate tier-specific operations
func demonstrateTierBehavior(cache core.CacheBackend) {
	// This would show how data moves between tiers
	// In production, you'd monitor this with metrics
}