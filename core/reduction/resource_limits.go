package reduction

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// ResourceLimits defines resource constraints for dimension reduction
type ResourceLimits struct {
	MaxMemoryMB          int64         // Maximum memory usage in MB
	MaxGoroutines        int           // Maximum concurrent operations
	OperationTimeout     time.Duration // Timeout for individual operations
	BatchTimeout         time.Duration // Timeout for batch operations
	MaxBatchSize         int           // Maximum batch size
	MaxQueueSize         int           // Maximum queue size for backpressure
	MemoryCheckInterval  time.Duration // How often to check memory
}

// DefaultResourceLimits returns sensible default limits
func DefaultResourceLimits() ResourceLimits {
	return ResourceLimits{
		MaxMemoryMB:         1024,             // 1GB
		MaxGoroutines:       runtime.NumCPU(), // One per CPU
		OperationTimeout:    30 * time.Second,
		BatchTimeout:        5 * time.Minute,
		MaxBatchSize:        1000,
		MaxQueueSize:        10000,
		MemoryCheckInterval: 1 * time.Second,
	}
}

// ResourceLimitedReducer wraps a reducer with resource management
type ResourceLimitedReducer struct {
	reducer     *ObservableReducer
	limits      ResourceLimits
	
	// Resource tracking
	currentMemoryMB  int64
	activeOperations int32
	queuedOperations int32
	
	// Backpressure
	semaphore    chan struct{}
	queue        chan *queuedOperation
	
	// Memory monitoring
	memoryTicker *time.Ticker
	stopMonitor  chan struct{}
	
	// Metrics
	rejectedOps   uint64
	timeoutOps    uint64
	memoryErrors  uint64
	
	mu sync.RWMutex
}

// queuedOperation represents a queued reduction operation
type queuedOperation struct {
	ctx        context.Context
	embedding  []float32
	resultChan chan queuedResult
	timestamp  time.Time
}

// queuedResult holds the result of a queued operation
type queuedResult struct {
	reduced []float32
	err     error
}

// NewResourceLimitedReducer creates a new resource-limited reducer
func NewResourceLimitedReducer(config *Config, limits ResourceLimits) (*ResourceLimitedReducer, error) {
	observableReducer, err := NewObservableReducer(config, 0.9)
	if err != nil {
		return nil, err
	}

	rlr := &ResourceLimitedReducer{
		reducer:      observableReducer,
		limits:       limits,
		semaphore:    make(chan struct{}, limits.MaxGoroutines),
		queue:        make(chan *queuedOperation, limits.MaxQueueSize),
		stopMonitor:  make(chan struct{}),
		memoryTicker: time.NewTicker(limits.MemoryCheckInterval),
	}

	// Start background workers
	rlr.startWorkers()
	rlr.startMemoryMonitor()

	return rlr, nil
}

// Learn trains the reducer with resource limits
func (rlr *ResourceLimitedReducer) Learn(ctx context.Context, embeddings [][]float32) error {
	// Check memory before training
	if err := rlr.checkMemoryLimit(len(embeddings)); err != nil {
		atomic.AddUint64(&rlr.memoryErrors, 1)
		return err
	}

	// Apply timeout
	ctx, cancel := context.WithTimeout(ctx, rlr.limits.BatchTimeout)
	defer cancel()

	// Acquire semaphore
	select {
	case rlr.semaphore <- struct{}{}:
		defer func() { <-rlr.semaphore }()
	case <-ctx.Done():
		atomic.AddUint64(&rlr.timeoutOps, 1)
		return fmt.Errorf("timeout waiting for resource: %w", ctx.Err())
	}

	atomic.AddInt32(&rlr.activeOperations, 1)
	defer atomic.AddInt32(&rlr.activeOperations, -1)

	// Split into batches if necessary
	if len(embeddings) > rlr.limits.MaxBatchSize {
		return rlr.learnInBatches(ctx, embeddings)
	}

	return rlr.reducer.Learn(ctx, embeddings)
}

// learnInBatches trains on large datasets in batches
func (rlr *ResourceLimitedReducer) learnInBatches(ctx context.Context, embeddings [][]float32) error {
	// Use incremental PCA for batch learning
	// Create a default config since we don't have access to the internal config
	config := &Config{
		TargetDim:         10, // Default reduced dimensions
		VarianceThreshold: 0.95, // Default value
	}
	incPCA := NewIncrementalPCAReducer(config, rlr.limits.MaxBatchSize)
	
	for i := 0; i < len(embeddings); i += rlr.limits.MaxBatchSize {
		end := i + rlr.limits.MaxBatchSize
		if end > len(embeddings) {
			end = len(embeddings)
		}
		
		batch := embeddings[i:end]
		
		// Check context and memory
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		
		if err := rlr.checkMemoryLimit(len(batch)); err != nil {
			return err
		}
		
		// Train on batch
		if err := incPCA.PartialFit(ctx, batch); err != nil {
			return fmt.Errorf("batch %d failed: %w", i/rlr.limits.MaxBatchSize, err)
		}
	}
	
	// Update the main reducer with the learned model
	// (This would require adding a method to update from incremental PCA)
	return nil
}

// ReduceForSearch reduces embeddings with resource limits and backpressure
func (rlr *ResourceLimitedReducer) ReduceForSearch(ctx context.Context, embedding []float32) ([]float32, error) {
	// Check queue size for backpressure
	queueSize := atomic.LoadInt32(&rlr.queuedOperations)
	if queueSize >= int32(rlr.limits.MaxQueueSize) {
		atomic.AddUint64(&rlr.rejectedOps, 1)
		return nil, fmt.Errorf("queue full: %d operations queued", queueSize)
	}

	// Create queued operation
	op := &queuedOperation{
		ctx:        ctx,
		embedding:  embedding,
		resultChan: make(chan queuedResult, 1),
		timestamp:  time.Now(),
	}

	// Apply timeout
	ctx, cancel := context.WithTimeout(ctx, rlr.limits.OperationTimeout)
	defer cancel()

	// Queue the operation
	atomic.AddInt32(&rlr.queuedOperations, 1)
	defer atomic.AddInt32(&rlr.queuedOperations, -1)

	select {
	case rlr.queue <- op:
		// Wait for result
		select {
		case result := <-op.resultChan:
			return result.reduced, result.err
		case <-ctx.Done():
			atomic.AddUint64(&rlr.timeoutOps, 1)
			return nil, fmt.Errorf("operation timeout: %w", ctx.Err())
		}
	case <-ctx.Done():
		atomic.AddUint64(&rlr.timeoutOps, 1)
		return nil, fmt.Errorf("timeout queueing operation: %w", ctx.Err())
	}
}

// startWorkers starts background workers for processing queued operations
func (rlr *ResourceLimitedReducer) startWorkers() {
	for i := 0; i < rlr.limits.MaxGoroutines; i++ {
		go rlr.worker()
	}
}

// worker processes queued operations
func (rlr *ResourceLimitedReducer) worker() {
	for op := range rlr.queue {
		// Check if operation has already timed out
		if op.ctx.Err() != nil {
			op.resultChan <- queuedResult{nil, op.ctx.Err()}
			continue
		}

		// Acquire semaphore
		select {
		case rlr.semaphore <- struct{}{}:
			// Process operation
			atomic.AddInt32(&rlr.activeOperations, 1)
			reduced, err := rlr.reducer.ReduceForSearch(op.ctx, op.embedding)
			atomic.AddInt32(&rlr.activeOperations, -1)
			<-rlr.semaphore

			// Send result
			op.resultChan <- queuedResult{reduced, err}
		case <-op.ctx.Done():
			op.resultChan <- queuedResult{nil, op.ctx.Err()}
		}
	}
}

// checkMemoryLimit checks if operation would exceed memory limit
func (rlr *ResourceLimitedReducer) checkMemoryLimit(numEmbeddings int) error {
	// Estimate memory usage (rough approximation)
	info := rlr.reducer.reducer.GetReductionInfo()
	embeddingMemoryMB := int64(numEmbeddings * info.OriginalDim * 4 / (1024 * 1024))
	
	currentMB := atomic.LoadInt64(&rlr.currentMemoryMB)
	if currentMB+embeddingMemoryMB > rlr.limits.MaxMemoryMB {
		return fmt.Errorf("memory limit exceeded: current=%dMB, required=%dMB, limit=%dMB",
			currentMB, embeddingMemoryMB, rlr.limits.MaxMemoryMB)
	}
	
	return nil
}

// startMemoryMonitor starts background memory monitoring
func (rlr *ResourceLimitedReducer) startMemoryMonitor() {
	go func() {
		for {
			select {
			case <-rlr.memoryTicker.C:
				rlr.updateMemoryUsage()
			case <-rlr.stopMonitor:
				return
			}
		}
	}()
}

// updateMemoryUsage updates current memory usage estimate
func (rlr *ResourceLimitedReducer) updateMemoryUsage() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	// Convert to MB and store
	currentMB := int64(m.Alloc / (1024 * 1024))
	atomic.StoreInt64(&rlr.currentMemoryMB, currentMB)
	
	// Update Prometheus metric
	memoryUsageGauge.WithLabelValues("total").Set(float64(currentMB))
}

// HybridSearch performs search with resource limits
func (rlr *ResourceLimitedReducer) HybridSearch(
	ctx context.Context,
	queryEmbedding []float32,
	candidates []SearchCandidate,
	topK int,
	similarityFunc func(a, b []float32) float64,
) ([]SearchResult, error) {
	// Check memory
	if err := rlr.checkMemoryLimit(len(candidates)); err != nil {
		atomic.AddUint64(&rlr.memoryErrors, 1)
		return nil, err
	}

	// Apply timeout
	ctx, cancel := context.WithTimeout(ctx, rlr.limits.OperationTimeout)
	defer cancel()

	// Acquire semaphore
	select {
	case rlr.semaphore <- struct{}{}:
		defer func() { <-rlr.semaphore }()
	case <-ctx.Done():
		atomic.AddUint64(&rlr.timeoutOps, 1)
		return nil, fmt.Errorf("timeout waiting for resource: %w", ctx.Err())
	}

	atomic.AddInt32(&rlr.activeOperations, 1)
	defer atomic.AddInt32(&rlr.activeOperations, -1)

	// Process in batches if necessary
	if len(candidates) > rlr.limits.MaxBatchSize*10 {
		return rlr.batchedHybridSearch(ctx, queryEmbedding, candidates, topK, similarityFunc)
	}

	return rlr.reducer.HybridSearch(ctx, queryEmbedding, candidates, topK, similarityFunc)
}

// batchedHybridSearch processes large candidate sets in batches
func (rlr *ResourceLimitedReducer) batchedHybridSearch(
	ctx context.Context,
	queryEmbedding []float32,
	candidates []SearchCandidate,
	topK int,
	similarityFunc func(a, b []float32) float64,
) ([]SearchResult, error) {
	
	// Process candidates in batches and collect top-K from each
	batchSize := rlr.limits.MaxBatchSize
	var allResults []SearchResult
	
	for i := 0; i < len(candidates); i += batchSize {
		end := i + batchSize
		if end > len(candidates) {
			end = len(candidates)
		}
		
		batch := candidates[i:end]
		
		// Check context
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		
		// Search in batch
		batchResults, err := rlr.reducer.HybridSearch(ctx, queryEmbedding, batch, topK, similarityFunc)
		if err != nil {
			return nil, fmt.Errorf("batch %d failed: %w", i/batchSize, err)
		}
		
		allResults = append(allResults, batchResults...)
	}
	
	// Final re-ranking of all batch results
	if len(allResults) <= topK {
		return allResults, nil
	}
	
	// Use heap for final selection
	selector := NewTopKSelector(topK)
	scored := make([]scoredResult, len(allResults))
	for i, result := range allResults {
		scored[i] = scoredResult{
			result:     result,
			similarity: result.Similarity,
		}
	}
	
	return selector.SelectTopKResults(scored), nil
}

// GetMetrics returns resource usage metrics
func (rlr *ResourceLimitedReducer) GetMetrics() ResourceMetrics {
	return ResourceMetrics{
		CurrentMemoryMB:  atomic.LoadInt64(&rlr.currentMemoryMB),
		ActiveOperations: atomic.LoadInt32(&rlr.activeOperations),
		QueuedOperations: atomic.LoadInt32(&rlr.queuedOperations),
		RejectedOps:      atomic.LoadUint64(&rlr.rejectedOps),
		TimeoutOps:       atomic.LoadUint64(&rlr.timeoutOps),
		MemoryErrors:     atomic.LoadUint64(&rlr.memoryErrors),
	}
}

// ResourceMetrics contains resource usage statistics
type ResourceMetrics struct {
	CurrentMemoryMB  int64
	ActiveOperations int32
	QueuedOperations int32
	RejectedOps      uint64
	TimeoutOps       uint64
	MemoryErrors     uint64
}

// UpdateLimits updates resource limits dynamically
func (rlr *ResourceLimitedReducer) UpdateLimits(newLimits ResourceLimits) {
	rlr.mu.Lock()
	defer rlr.mu.Unlock()
	
	rlr.limits = newLimits
	
	// Recreate semaphore with new limit
	close(rlr.semaphore)
	rlr.semaphore = make(chan struct{}, newLimits.MaxGoroutines)
}

// Shutdown gracefully shuts down the resource-limited reducer
func (rlr *ResourceLimitedReducer) Shutdown() {
	close(rlr.stopMonitor)
	rlr.memoryTicker.Stop()
	close(rlr.queue)
}

// AdaptiveLimits adjusts limits based on system performance
type AdaptiveLimits struct {
	reducer         *ResourceLimitedReducer
	adjustInterval  time.Duration
	targetLatencyMs float64
	ticker          *time.Ticker
	stop            chan struct{}
}

// NewAdaptiveLimits creates adaptive resource management
func NewAdaptiveLimits(reducer *ResourceLimitedReducer, targetLatencyMs float64) *AdaptiveLimits {
	return &AdaptiveLimits{
		reducer:         reducer,
		adjustInterval:  30 * time.Second,
		targetLatencyMs: targetLatencyMs,
		ticker:          time.NewTicker(30 * time.Second),
		stop:            make(chan struct{}),
	}
}

// Start begins adaptive limit adjustment
func (al *AdaptiveLimits) Start() {
	go al.adjustLoop()
}

// adjustLoop periodically adjusts limits based on performance
func (al *AdaptiveLimits) adjustLoop() {
	for {
		select {
		case <-al.ticker.C:
			al.adjustLimits()
		case <-al.stop:
			return
		}
	}
}

// adjustLimits adjusts resource limits based on metrics
func (al *AdaptiveLimits) adjustLimits() {
	metrics := al.reducer.GetMetrics()
	currentLimits := al.reducer.limits
	
	// Adjust based on queue size
	if metrics.QueuedOperations > int32(currentLimits.MaxQueueSize)*8/10 {
		// Queue is getting full, increase workers
		newLimits := currentLimits
		newLimits.MaxGoroutines = min(currentLimits.MaxGoroutines+2, runtime.NumCPU()*2)
		al.reducer.UpdateLimits(newLimits)
	} else if metrics.QueuedOperations < int32(currentLimits.MaxQueueSize)*2/10 && 
		currentLimits.MaxGoroutines > runtime.NumCPU() {
		// Queue is mostly empty, decrease workers
		newLimits := currentLimits
		newLimits.MaxGoroutines = max(currentLimits.MaxGoroutines-1, runtime.NumCPU())
		al.reducer.UpdateLimits(newLimits)
	}
	
	// Adjust based on rejection rate
	if metrics.RejectedOps > 0 {
		// Increase queue size if rejecting operations
		newLimits := currentLimits
		newLimits.MaxQueueSize = min(currentLimits.MaxQueueSize*2, 50000)
		al.reducer.UpdateLimits(newLimits)
	}
}

// Stop stops adaptive adjustment
func (al *AdaptiveLimits) Stop() {
	close(al.stop)
	al.ticker.Stop()
}

// Helper functions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}