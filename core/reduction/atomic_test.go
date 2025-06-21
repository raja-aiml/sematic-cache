package reduction

import (
	"context"
	"math"
	"sync"
	"sync/atomic"
	"testing"
)

// TestAtomicFloatOperations tests the atomic float64 operations
func TestAtomicFloatOperations(t *testing.T) {
	// Test atomic float64 store and load
	var atomicFloat uint64

	// Store a float64 value
	testValue := 3.14159
	atomic.StoreUint64(&atomicFloat, math.Float64bits(testValue))

	// Load and verify
	loadedValue := math.Float64frombits(atomic.LoadUint64(&atomicFloat))
	if loadedValue != testValue {
		t.Errorf("Atomic float operation failed: got %f, want %f", loadedValue, testValue)
	}
}

// TestConcurrentMetricsUpdates tests concurrent updates to metrics
func TestConcurrentMetricsUpdates(t *testing.T) {
	config := &Config{
		TargetDim:         50,
		VarianceThreshold: 0.95,
	}

	reducer, err := NewDimensionReducer(config)
	if err != nil {
		t.Fatalf("Failed to create reducer: %v", err)
	}

	// Generate test embeddings
	embeddings := generateTestEmbeddings(100, 128)

	ctx := context.Background()
	err = reducer.Learn(ctx, embeddings)
	if err != nil {
		t.Fatalf("Failed to learn: %v", err)
	}

	// Test concurrent updates
	var wg sync.WaitGroup
	numGoroutines := 100
	numOperations := 1000

	// Concurrent reductions
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				_, _ = reducer.ReduceForSearch(ctx, embeddings[j%len(embeddings)])
			}
		}()
	}

	// Concurrent hit rate updates
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				before := 0.5 + float64(id*j%20)/100.0
				after := 0.4 + float64(id*j%20)/100.0
				reducer.UpdateHitRates(before, after)
			}
		}(i)
	}

	// Concurrent metrics reads
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				_ = reducer.GetMetrics()
			}
		}()
	}

	wg.Wait()

	// Verify final metrics
	metrics := reducer.GetMetrics()
	expectedReductions := int64(numGoroutines * numOperations)

	if metrics.ReducedDimQueries != expectedReductions {
		t.Errorf("ReducedDimQueries = %d, want %d", metrics.ReducedDimQueries, expectedReductions)
	}

	// Verify hit rates were updated (last update wins)
	if metrics.HitRateBeforeReduction == 0 || metrics.HitRateAfterReduction == 0 {
		t.Error("Hit rates were not updated")
	}
}

// TestABTestConcurrentImpressions tests concurrent impression recording
func TestABTestConcurrentImpressions(t *testing.T) {
	manager := NewABTestManager(Strategy{
		ID:        "default",
		Name:      "Default",
		TargetDim: 100,
	})

	strategies := []Strategy{
		{ID: "s1", Name: "Strategy 1", TargetDim: 50},
		{ID: "s2", Name: "Strategy 2", TargetDim: 100},
	}

	config := ABTestConfig{
		MinImpressions:        10000,
		MinDurationHours:      1,
		ConfidenceLevel:       0.95,
		SignificanceThreshold: 0.05,
	}

	test, err := manager.CreateTest(config, strategies, []float64{0.5, 0.5})
	if err != nil {
		t.Fatalf("Failed to create test: %v", err)
	}

	err = manager.StartTest(test.ID)
	if err != nil {
		t.Fatalf("Failed to start test: %v", err)
	}

	// Concurrent impressions
	var wg sync.WaitGroup
	ctx := context.Background()
	numGoroutines := 100
	numImpressions := 100

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numImpressions; j++ {
				// Simulate impressions for both strategies
				for _, strategy := range strategies {
					metrics := ImpressionMetrics{
						CacheHit:        j%2 == 0,
						LatencyMs:       int64(10 + j%5),
						SearchLatencyMs: int64(5 + j%3),
						SimilarityScore: 0.8 + float64(j%10)/100.0,
					}
					manager.RecordImpression(ctx, strategy.ID, metrics)
				}
			}
		}(i)
	}

	wg.Wait()

	// Verify results
	expectedImpressions := int64(numGoroutines * numImpressions)

	for _, strategy := range strategies {
		results := test.Results[strategy.ID]
		impressions := atomic.LoadInt64(&results.Impressions)

		if impressions != expectedImpressions {
			t.Errorf("Strategy %s: impressions = %d, want %d", strategy.ID, impressions, expectedImpressions)
		}

		// Verify similarity scores were recorded
		count := atomic.LoadInt64(&results.SimilarityScoreCount)
		if count != expectedImpressions {
			t.Errorf("Strategy %s: similarity count = %d, want %d", strategy.ID, count, expectedImpressions)
		}

		// Verify average is reasonable
		avgScore := math.Float64frombits(atomic.LoadUint64(&results.AvgSimilarityScore))
		if avgScore < 0.8 || avgScore > 0.9 {
			t.Errorf("Strategy %s: average similarity score %f is out of expected range [0.8, 0.9]", strategy.ID, avgScore)
		}
	}
}

// TestMemoryUsageAtomicOperations tests atomic operations on memory usage
func TestMemoryUsageAtomicOperations(t *testing.T) {
	results := &TestResults{}

	// Test concurrent memory updates
	var wg sync.WaitGroup
	numGoroutines := 100

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				memoryMB := float64(id*100 + j)
				results.SetMemoryUsedMB(memoryMB)
			}
		}(i)
	}

	wg.Wait()

	// Final value should be from one of the last updates
	finalMemory := results.GetMemoryUsedMB()
	if finalMemory < 0 || finalMemory > 10000 {
		t.Errorf("Unexpected final memory value: %f", finalMemory)
	}
}
