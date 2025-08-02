package reduction

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"
)

// TestObservableReducer tests the observable reducer with tracing and metrics
func TestObservableReducer(t *testing.T) {
	config := &Config{
		TargetDim:         2,
		VarianceThreshold: 0,
	}

	reducer, err := NewObservableReducer(config, 0.9)
	if err != nil {
		t.Fatalf("Failed to create observable reducer: %v", err)
	}

	ctx := context.Background()

	// Test Learn with observability
	embeddings := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 3.0, 4.0, 5.0},
		{3.0, 4.0, 5.0, 6.0},
		{4.0, 5.0, 6.0, 7.0},
	}

	err = reducer.Learn(ctx, embeddings)
	if err != nil {
		t.Fatalf("Failed to learn: %v", err)
	}

	// Test ReduceForSearch
	query := []float32{3.0, 4.0, 5.0, 6.0}
	reduced, err := reducer.ReduceForSearch(ctx, query)
	if err != nil {
		t.Fatalf("Failed to reduce: %v", err)
	}

	if len(reduced) != 2 {
		t.Errorf("Expected reduced dim 2, got %d", len(reduced))
	}

	// Test HybridSearch
	candidates := []SearchCandidate{
		{ID: "1", Embedding: embeddings[0], ReducedEmbedding: []float32{1.0, 2.0}},
		{ID: "2", Embedding: embeddings[1], ReducedEmbedding: []float32{2.0, 3.0}},
	}

	results, err := reducer.HybridSearch(ctx, query, candidates, 1, func(a, b []float32) float64 {
		var sum float64
		for i := range a {
			sum += float64(a[i]) * float64(b[i])
		}
		return sum
	})

	if err != nil {
		t.Fatalf("Failed hybrid search: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}

	// Test health check
	err = reducer.HealthCheck(ctx)
	if err != nil {
		t.Errorf("Health check failed: %v", err)
	}

	// Test degradation callback
	reducer.SetDegradationCallback(func(ctx context.Context, metrics *QualityMetrics) {
		accuracy := getFloat64FromAtomic(&metrics.accuracyScore)
		t.Logf("Degradation detected: accuracy=%v", accuracy)
	})

	// Force check degradation (won't trigger unless accuracy is low)
	reducer.checkQualityDegradation(ctx)
}

// TestIncrementalPCA tests incremental PCA functionality
func TestIncrementalPCA(t *testing.T) {
	config := &Config{
		TargetDim:         2,
		VarianceThreshold: 0,
	}

	incPCA := NewIncrementalPCAReducer(config, 2)
	ctx := context.Background()

	// Test with initial batch
	batch1 := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 3.0, 4.0, 5.0},
	}

	err := incPCA.PartialFit(ctx, batch1)
	if err != nil {
		t.Fatalf("Failed partial fit: %v", err)
	}

	if !incPCA.IsInitialized() {
		t.Error("Expected PCA to be initialized")
	}

	if incPCA.GetNSamplesSeen() != 2 {
		t.Errorf("Expected 2 samples seen, got %d", incPCA.GetNSamplesSeen())
	}

	// Test with second batch
	batch2 := [][]float32{
		{3.0, 4.0, 5.0, 6.0},
		{4.0, 5.0, 6.0, 7.0},
	}

	err = incPCA.PartialFit(ctx, batch2)
	if err != nil {
		t.Fatalf("Failed second partial fit: %v", err)
	}

	if incPCA.GetNSamplesSeen() != 4 {
		t.Errorf("Expected 4 samples seen, got %d", incPCA.GetNSamplesSeen())
	}

	// Test transform
	testData := [][]float32{{2.5, 3.5, 4.5, 5.5}}
	reduced, err := incPCA.Transform(ctx, testData)
	if err != nil {
		t.Fatalf("Failed transform: %v", err)
	}

	if len(reduced[0]) != 2 {
		t.Errorf("Expected reduced dim 2, got %d", len(reduced[0]))
	}

	// Test drift detection
	driftBatch := [][]float32{
		{10.0, 20.0, 30.0, 40.0}, // Very different from training data
		{11.0, 21.0, 31.0, 41.0},
	}

	shouldUpdate, drift := incPCA.ShouldUpdate(driftBatch, 5.0)
	if !shouldUpdate {
		t.Errorf("Expected drift detection to trigger update, drift=%f", drift)
	}

	// Test adaptive batch size
	adaptiveSize := incPCA.AdaptiveBatchSize()
	if adaptiveSize != 2 { // Still under 1000 samples
		t.Errorf("Expected batch size 2, got %d", adaptiveSize)
	}
}

// TestResourceLimits tests resource limiting functionality
func TestResourceLimits(t *testing.T) {
	config := &Config{
		TargetDim:         2,
		VarianceThreshold: 0,
	}

	limits := ResourceLimits{
		MaxMemoryMB:         100,
		MaxGoroutines:       2,
		OperationTimeout:    1 * time.Second,
		BatchTimeout:        5 * time.Second,
		MaxBatchSize:        10,
		MaxQueueSize:        20,
		MemoryCheckInterval: 100 * time.Millisecond,
	}

	reducer, err := NewResourceLimitedReducer(config, limits)
	if err != nil {
		t.Fatalf("Failed to create resource limited reducer: %v", err)
	}
	defer reducer.Shutdown()

	ctx := context.Background()

	// Train with small batch
	embeddings := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 3.0, 4.0, 5.0},
		{3.0, 4.0, 5.0, 6.0},
		{4.0, 5.0, 6.0, 7.0},
	}

	err = reducer.Learn(ctx, embeddings)
	if err != nil {
		t.Fatalf("Failed to learn: %v", err)
	}

	// Test concurrent operations
	var wg sync.WaitGroup
	errors := make([]error, 10)

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			query := []float32{float32(idx), float32(idx + 1), float32(idx + 2), float32(idx + 3)}
			_, err := reducer.ReduceForSearch(ctx, query)
			errors[idx] = err
		}(i)
	}

	wg.Wait()

	// Check that operations succeeded (with resource limits)
	successCount := 0
	for _, err := range errors {
		if err == nil {
			successCount++
		}
	}

	if successCount == 0 {
		t.Error("Expected at least some operations to succeed")
	}

	// Test metrics
	metrics := reducer.GetMetrics()
	t.Logf("Resource metrics: memory=%dMB, active=%d, queued=%d, rejected=%d",
		metrics.CurrentMemoryMB, metrics.ActiveOperations, metrics.QueuedOperations, metrics.RejectedOps)

	// Test timeout
	slowCtx, cancel := context.WithTimeout(ctx, 10*time.Millisecond)
	defer cancel()

	time.Sleep(20 * time.Millisecond) // Ensure timeout
	_, err = reducer.ReduceForSearch(slowCtx, []float32{1.0, 2.0, 3.0, 4.0})
	if err == nil {
		t.Error("Expected timeout error")
	}
}

// TestBackpressure tests backpressure handling
func TestBackpressure(t *testing.T) {
	config := &Config{
		TargetDim: 2,
	}

	limits := ResourceLimits{
		MaxMemoryMB:         100,
		MaxGoroutines:       1, // Single worker to force queueing
		OperationTimeout:    100 * time.Millisecond,
		BatchTimeout:        1 * time.Second,
		MaxBatchSize:        10,
		MaxQueueSize:        5, // Small queue to test backpressure
		MemoryCheckInterval: 100 * time.Millisecond,
	}

	reducer, err := NewResourceLimitedReducer(config, limits)
	if err != nil {
		t.Fatalf("Failed to create reducer: %v", err)
	}
	defer reducer.Shutdown()

	// Train first
	embeddings := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 3.0, 4.0, 5.0},
	}
	reducer.Learn(context.Background(), embeddings)

	// Flood with requests to trigger backpressure
	ctx := context.Background()
	var wg sync.WaitGroup
	rejectedCount := 0
	mu := sync.Mutex{}

	for i := 0; i < 20; i++ { // More than queue size
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			query := []float32{1.0, 2.0, 3.0, 4.0}
			_, err := reducer.ReduceForSearch(ctx, query)
			if err != nil && fmt.Sprintf("%v", err) == fmt.Sprintf("queue full: %d operations queued", limits.MaxQueueSize) {
				mu.Lock()
				rejectedCount++
				mu.Unlock()
			}
		}(i)
	}

	wg.Wait()

	if rejectedCount == 0 {
		t.Error("Expected some operations to be rejected due to backpressure")
	}

	t.Logf("Rejected %d operations due to backpressure", rejectedCount)
}

// TestAdaptiveLimits tests adaptive resource management
func TestAdaptiveLimits(t *testing.T) {
	config := &Config{
		TargetDim: 2,
	}

	limits := DefaultResourceLimits()
	limits.MaxGoroutines = 2

	reducer, err := NewResourceLimitedReducer(config, limits)
	if err != nil {
		t.Fatalf("Failed to create reducer: %v", err)
	}
	defer reducer.Shutdown()

	// Create adaptive limits manager
	adaptive := NewAdaptiveLimits(reducer, 10.0)     // 10ms target latency
	adaptive.adjustInterval = 100 * time.Millisecond // Fast for testing
	adaptive.Start()
	defer adaptive.Stop()

	// Train
	embeddings := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 3.0, 4.0, 5.0},
	}
	reducer.Learn(context.Background(), embeddings)

	// Generate load
	ctx := context.Background()
	for i := 0; i < 100; i++ {
		go func() {
			query := []float32{1.0, 2.0, 3.0, 4.0}
			reducer.ReduceForSearch(ctx, query)
		}()
	}

	// Wait for adaptive adjustment
	time.Sleep(200 * time.Millisecond)

	// Check if limits were adjusted
	newLimits := reducer.limits
	t.Logf("Adaptive limits: goroutines=%d, queue=%d",
		newLimits.MaxGoroutines, newLimits.MaxQueueSize)
}

// TestHealthCheckFailure tests health check failure scenarios
func TestHealthCheckFailure(t *testing.T) {
	config := &Config{
		TargetDim:         2,
		VarianceThreshold: 0.95, // High threshold
	}

	reducer, err := NewObservableReducer(config, 0.9)
	if err != nil {
		t.Fatalf("Failed to create reducer: %v", err)
	}

	ctx := context.Background()

	// Health check should fail when not trained
	err = reducer.HealthCheck(ctx)
	if err == nil {
		t.Error("Expected health check to fail when not trained")
	}

	// Train with data that won't meet variance threshold
	embeddings := [][]float32{
		{1.0, 1.0, 1.0, 1.0}, // Low variance data
		{1.1, 1.1, 1.1, 1.1},
		{1.0, 1.0, 1.0, 1.0},
		{1.1, 1.1, 1.1, 1.1},
	}

	reducer.Learn(ctx, embeddings)

	// Health check might fail due to low variance
	err = reducer.HealthCheck(ctx)
	t.Logf("Health check result: %v", err)
}
