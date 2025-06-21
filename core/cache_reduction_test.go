package core

import (
	"context"
	"testing"

	"github.com/raja-aiml/sematic-cache/core/reduction"
)

func TestCache_WithDimensionReduction(t *testing.T) {
	// Create test embeddings
	embedFunc := func(text string) ([]float32, error) {
		// Simple hash-based embedding for testing
		h := 0
		for _, c := range text {
			h = h*31 + int(c)
		}

		emb := make([]float32, 100)
		for i := range emb {
			emb[i] = float32((h+i)%100) / 100.0
		}
		return emb, nil
	}

	// Create dimension reducer
	config := &reduction.Config{
		TargetDim: 20,
	}

	reducer, err := reduction.NewDimensionReducer(config)
	if err != nil {
		t.Fatalf("Failed to create reducer: %v", err)
	}

	// Create cache with dimension reduction
	cache, err := NewCache(100,
		WithEmbeddingFunc(embedFunc),
		WithDimensionReduction(reducer),
		WithMinSimilarity(0.7),
	)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}

	// Test that ensureDualEmbeddings is set
	if !cache.ensureDualEmbeddings {
		t.Error("ensureDualEmbeddings should be true when dimension reduction is enabled")
	}

	// Add some entries
	testData := []struct {
		prompt string
		answer string
	}{
		{"What is AI?", "Artificial Intelligence"},
		{"What is ML?", "Machine Learning"},
		{"What is DL?", "Deep Learning"},
		{"What is NLP?", "Natural Language Processing"},
		{"What is CV?", "Computer Vision"},
	}

	for _, td := range testData {
		if err := cache.SetPrompt(td.prompt, td.answer); err != nil {
			t.Errorf("Failed to set %s: %v", td.prompt, err)
		}
	}

	// Entries should not have reduced embeddings yet (reducer not trained)
	stats := cache.GetDualEmbeddingStats()
	if stats.WithReducedEmbedding > 0 {
		t.Error("Entries should not have reduced embeddings before training")
	}

	// Train dimension reducer
	ctx := context.Background()
	if err := cache.TrainDimensionReducer(ctx); err != nil {
		t.Fatalf("Failed to train reducer: %v", err)
	}

	// Now all entries should have reduced embeddings
	stats = cache.GetDualEmbeddingStats()
	if stats.WithBothEmbeddings != len(testData) {
		t.Errorf("Expected %d entries with both embeddings, got %d",
			len(testData), stats.WithBothEmbeddings)
	}

	// Test that HasDimensionReduction returns true
	if !cache.HasDimensionReduction() {
		t.Error("HasDimensionReduction should return true after training")
	}

	// Add new entry after training - should get both embeddings
	if err := cache.SetPrompt("What is RL?", "Reinforcement Learning"); err != nil {
		t.Errorf("Failed to set new entry: %v", err)
	}

	// Check the new entry has both embeddings
	if err := cache.EnsureBothEmbeddings("What is RL?"); err != nil {
		t.Errorf("New entry should have both embeddings: %v", err)
	}
}

func TestCache_HybridSearch(t *testing.T) {
	// Create test embeddings with some structure
	embedFunc := func(text string) ([]float32, error) {
		// Create embeddings that have similarity based on first letter
		firstChar := text[0]
		base := float32(firstChar-'A') / 26.0

		emb := make([]float32, 50)
		for i := range emb {
			emb[i] = base + float32(i)*0.01
			if i%10 == 0 {
				emb[i] += float32(len(text)) * 0.01
			}
		}
		return emb, nil
	}

	// Create reducer
	config := &reduction.Config{
		TargetDim: 10,
	}

	reducer, err := reduction.NewDimensionReducer(config)
	if err != nil {
		t.Fatalf("Failed to create reducer: %v", err)
	}

	// Create cache
	cache, err := NewCache(100,
		WithEmbeddingFunc(embedFunc),
		WithDimensionReduction(reducer),
		WithMinSimilarity(0.5),
	)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}

	// Add entries
	entries := []struct {
		prompt string
		answer string
	}{
		{"Apple", "A fruit"},
		{"Apricot", "Another fruit"},
		{"Banana", "Yellow fruit"},
		{"Blueberry", "Blue fruit"},
		{"Cherry", "Red fruit"},
		{"Date", "Sweet fruit"},
	}

	for _, e := range entries {
		if err := cache.SetPrompt(e.prompt, e.answer); err != nil {
			t.Fatalf("Failed to add %s: %v", e.prompt, err)
		}
	}

	// Train reducer
	ctx := context.Background()
	if err := cache.TrainDimensionReducer(ctx); err != nil {
		t.Fatalf("Failed to train reducer: %v", err)
	}

	// Test hybrid search
	queryEmb, _ := embedFunc("Avocado")
	results := cache.GetTopKByEmbedding(queryEmb, 3)

	if len(results) == 0 {
		t.Fatal("Expected some results from hybrid search")
	}

	// Results should be sorted by similarity
	for i := 1; i < len(results); i++ {
		if results[i].Similarity > results[i-1].Similarity {
			t.Error("Results not sorted by similarity")
		}
	}

	// "Apple" and "Apricot" should be in top results (same first letter)
	foundApple := false
	foundApricot := false
	for _, r := range results {
		if r.Prompt == "Apple" {
			foundApple = true
		}
		if r.Prompt == "Apricot" {
			foundApricot = true
		}
	}

	if !foundApple || !foundApricot {
		t.Error("Expected Apple and Apricot in top results")
	}
}

func TestCache_EnsureReducedEmbeddings(t *testing.T) {
	embedFunc := func(text string) ([]float32, error) {
		emb := make([]float32, 30)
		for i := range emb {
			emb[i] = float32(len(text)+i) / 30.0
		}
		return emb, nil
	}

	// Create cache without reducer initially
	cache, err := NewCache(50,
		WithEmbeddingFunc(embedFunc),
	)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}

	// Add entries
	for i := 0; i < 15; i++ {
		prompt := string(rune('A' + i))
		answer := "Answer " + prompt
		cache.SetPrompt(prompt, answer)
	}

	// Now add reducer
	config := &reduction.Config{
		TargetDim: 10,
	}

	reducer, _ := reduction.NewDimensionReducer(config)
	cache.dimensionReducer = reducer

	// Try to ensure reduced embeddings - should auto-train
	ctx := context.Background()
	err = cache.EnsureReducedEmbeddings(ctx)
	if err != nil {
		t.Fatalf("EnsureReducedEmbeddings failed: %v", err)
	}

	// Check all entries have reduced embeddings
	stats := cache.GetDualEmbeddingStats()
	if stats.WithBothEmbeddings != stats.TotalEntries {
		t.Errorf("Expected all %d entries to have both embeddings, got %d",
			stats.TotalEntries, stats.WithBothEmbeddings)
	}
}

func TestCache_DimensionReductionMetrics(t *testing.T) {
	embedFunc := func(text string) ([]float32, error) {
		emb := make([]float32, 40)
		for i := range emb {
			emb[i] = float32(i) / 40.0
		}
		return emb, nil
	}

	config := &reduction.Config{
		TargetDim: 10,
	}

	reducer, _ := reduction.NewDimensionReducer(config)

	cache, err := NewCache(100,
		WithEmbeddingFunc(embedFunc),
		WithDimensionReduction(reducer),
	)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}

	// Add entries and train
	for i := 0; i < 20; i++ {
		cache.SetPrompt(string(rune('A'+i)), "Answer")
	}

	ctx := context.Background()
	cache.TrainDimensionReducer(ctx)

	// Update hit rates
	cache.UpdateReductionHitRates(0.8, 0.75)

	// Get metrics
	metrics := cache.GetDimensionReductionMetrics()
	if metrics == nil {
		t.Fatal("Expected dimension reduction metrics")
	}

	// Just check that metrics exist
	// The fields are unexported, so we just check that metrics is not nil
	// and that the reducer has the expected info
	info := cache.dimensionReducer.GetReductionInfo()
	if info.VarianceExplained == 0 {
		t.Error("Expected variance explained to be set")
	}
}

func TestCache_RegularSearchFallback(t *testing.T) {
	embedFunc := func(text string) ([]float32, error) {
		emb := make([]float32, 20)
		for i := range emb {
			emb[i] = float32(len(text)) / 20.0
		}
		return emb, nil
	}

	// Create cache with reducer that's not trained
	config := &reduction.Config{
		TargetDim: 5,
	}

	reducer, _ := reduction.NewDimensionReducer(config)

	cache, err := NewCache(50,
		WithEmbeddingFunc(embedFunc),
		WithDimensionReduction(reducer),
		WithMinSimilarity(0.5),
	)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}

	// Add entries
	cache.SetPrompt("Short", "Short answer")
	cache.SetPrompt("Medium text", "Medium answer")
	cache.SetPrompt("Very long text here", "Long answer")

	// Search should fall back to regular search since reducer isn't trained
	queryEmb, _ := embedFunc("Test query")
	results := cache.GetTopKByEmbedding(queryEmb, 2)

	// Should still get results
	if len(results) == 0 {
		t.Error("Expected results from fallback search")
	}
}
