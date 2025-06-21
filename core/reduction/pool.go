package reduction

import (
	"sync"
)

// ObjectPools manages pools for frequently allocated objects
type ObjectPools struct {
	candidatePool    sync.Pool
	resultPool       sync.Pool
	embeddingPool    sync.Pool
	float32SlicePool sync.Pool
	scoredCandPool   sync.Pool
	scoredResultPool sync.Pool
}

// Global pools instance
var pools = &ObjectPools{}

func init() {
	// Initialize pools with factory functions
	pools.candidatePool = sync.Pool{
		New: func() interface{} {
			return &SearchCandidate{
				Metadata: make(map[string]interface{}),
			}
		},
	}

	pools.resultPool = sync.Pool{
		New: func() interface{} {
			return &SearchResult{}
		},
	}

	pools.embeddingPool = sync.Pool{
		New: func() interface{} {
			// Default to 1536 dimensions (OpenAI embedding size)
			return make([]float32, 0, 1536)
		},
	}

	pools.float32SlicePool = sync.Pool{
		New: func() interface{} {
			// Default to reduced dimension size
			return make([]float32, 0, 384)
		},
	}

	pools.scoredCandPool = sync.Pool{
		New: func() interface{} {
			return &scoredCandidate{}
		},
	}

	pools.scoredResultPool = sync.Pool{
		New: func() interface{} {
			return &scoredResult{}
		},
	}
}

// GetCandidate gets a SearchCandidate from the pool
func GetCandidate() *SearchCandidate {
	candidate := pools.candidatePool.Get().(*SearchCandidate)
	// Reset fields
	candidate.ID = ""
	candidate.Embedding = candidate.Embedding[:0]
	candidate.ReducedEmbedding = candidate.ReducedEmbedding[:0]
	// Clear metadata
	for k := range candidate.Metadata {
		delete(candidate.Metadata, k)
	}
	return candidate
}

// PutCandidate returns a SearchCandidate to the pool
func PutCandidate(c *SearchCandidate) {
	if c == nil {
		return
	}
	pools.candidatePool.Put(c)
}

// GetResult gets a SearchResult from the pool
func GetResult() *SearchResult {
	return pools.resultPool.Get().(*SearchResult)
}

// PutResult returns a SearchResult to the pool
func PutResult(r *SearchResult) {
	if r == nil {
		return
	}
	pools.resultPool.Put(r)
}

// GetEmbedding gets an embedding slice from the pool
func GetEmbedding(size int) []float32 {
	if size <= 384 {
		slice := pools.float32SlicePool.Get().([]float32)
		return slice[:0]
	}

	slice := pools.embeddingPool.Get().([]float32)
	if cap(slice) < size {
		return make([]float32, 0, size)
	}
	return slice[:0]
}

// PutEmbedding returns an embedding slice to the pool
func PutEmbedding(e []float32) {
	if e == nil {
		return
	}

	// Reset slice but keep capacity
	e = e[:0]

	if cap(e) <= 384 {
		pools.float32SlicePool.Put(e)
	} else if cap(e) <= 1536 {
		pools.embeddingPool.Put(e)
	}
	// Don't pool very large slices
}

// GetScoredCandidate gets a scoredCandidate from the pool
func GetScoredCandidate() *scoredCandidate {
	return pools.scoredCandPool.Get().(*scoredCandidate)
}

// PutScoredCandidate returns a scoredCandidate to the pool
func PutScoredCandidate(s *scoredCandidate) {
	if s == nil {
		return
	}
	pools.scoredCandPool.Put(s)
}

// GetScoredResult gets a scoredResult from the pool
func GetScoredResult() *scoredResult {
	return pools.scoredResultPool.Get().(*scoredResult)
}

// PutScoredResult returns a scoredResult to the pool
func PutScoredResult(s *scoredResult) {
	if s == nil {
		return
	}
	pools.scoredResultPool.Put(s)
}

// CandidateSlicePool manages pools of SearchCandidate slices
type CandidateSlicePool struct {
	pools map[int]*sync.Pool
	mu    sync.RWMutex
}

// Global candidate slice pool
var candidateSlicePool = &CandidateSlicePool{
	pools: make(map[int]*sync.Pool),
}

// GetCandidateSlice gets a candidate slice of specified capacity
func GetCandidateSlice(cap int) []SearchCandidate {
	// Round up to nearest power of 2 for better pooling
	poolCap := nextPowerOf2(cap)

	candidateSlicePool.mu.RLock()
	pool, exists := candidateSlicePool.pools[poolCap]
	candidateSlicePool.mu.RUnlock()

	if !exists {
		candidateSlicePool.mu.Lock()
		pool, exists = candidateSlicePool.pools[poolCap]
		if !exists {
			pool = &sync.Pool{
				New: func() interface{} {
					return make([]SearchCandidate, 0, poolCap)
				},
			}
			candidateSlicePool.pools[poolCap] = pool
		}
		candidateSlicePool.mu.Unlock()
	}

	slice := pool.Get().([]SearchCandidate)
	return slice[:0]
}

// PutCandidateSlice returns a candidate slice to the pool
func PutCandidateSlice(s []SearchCandidate) {
	if s == nil {
		return
	}

	poolCap := nextPowerOf2(cap(s))

	candidateSlicePool.mu.RLock()
	pool, exists := candidateSlicePool.pools[poolCap]
	candidateSlicePool.mu.RUnlock()

	if exists {
		// Clear the slice before returning to pool
		for i := range s {
			s[i] = SearchCandidate{}
		}
		pool.Put(s[:0])
	}
}

// ResultSlicePool manages pools of SearchResult slices
type ResultSlicePool struct {
	pools map[int]*sync.Pool
	mu    sync.RWMutex
}

// Global result slice pool
var resultSlicePool = &ResultSlicePool{
	pools: make(map[int]*sync.Pool),
}

// GetResultSlice gets a result slice of specified capacity
func GetResultSlice(cap int) []SearchResult {
	// Round up to nearest power of 2 for better pooling
	poolCap := nextPowerOf2(cap)

	resultSlicePool.mu.RLock()
	pool, exists := resultSlicePool.pools[poolCap]
	resultSlicePool.mu.RUnlock()

	if !exists {
		resultSlicePool.mu.Lock()
		pool, exists = resultSlicePool.pools[poolCap]
		if !exists {
			pool = &sync.Pool{
				New: func() interface{} {
					return make([]SearchResult, 0, poolCap)
				},
			}
			resultSlicePool.pools[poolCap] = pool
		}
		resultSlicePool.mu.Unlock()
	}

	slice := pool.Get().([]SearchResult)
	return slice[:0]
}

// PutResultSlice returns a result slice to the pool
func PutResultSlice(s []SearchResult) {
	if s == nil {
		return
	}

	poolCap := nextPowerOf2(cap(s))

	resultSlicePool.mu.RLock()
	pool, exists := resultSlicePool.pools[poolCap]
	resultSlicePool.mu.RUnlock()

	if exists {
		// Clear the slice before returning to pool
		for i := range s {
			s[i] = SearchResult{}
		}
		pool.Put(s[:0])
	}
}

// nextPowerOf2 returns the next power of 2 >= n
func nextPowerOf2(n int) int {
	if n <= 0 {
		return 1
	}

	// Handle case where n is already a power of 2
	if n&(n-1) == 0 {
		return n
	}

	// Find the next power of 2
	power := 1
	for power < n {
		power <<= 1
	}
	return power
}

// EmbeddingBatchPool manages pools for batch processing
type EmbeddingBatchPool struct {
	pool sync.Pool
}

// Global embedding batch pool
var embeddingBatchPool = &EmbeddingBatchPool{
	pool: sync.Pool{
		New: func() interface{} {
			return &EmbeddingBatch{
				Embeddings: make([][]float32, 0, 100),
			}
		},
	},
}

// EmbeddingBatch represents a batch of embeddings for processing
type EmbeddingBatch struct {
	Embeddings [][]float32
}

// GetEmbeddingBatch gets an embedding batch from the pool
func GetEmbeddingBatch() *EmbeddingBatch {
	batch := embeddingBatchPool.pool.Get().(*EmbeddingBatch)
	batch.Embeddings = batch.Embeddings[:0]
	return batch
}

// PutEmbeddingBatch returns an embedding batch to the pool
func PutEmbeddingBatch(b *EmbeddingBatch) {
	if b == nil {
		return
	}

	// Clear embeddings
	for i := range b.Embeddings {
		b.Embeddings[i] = nil
	}
	b.Embeddings = b.Embeddings[:0]

	embeddingBatchPool.pool.Put(b)
}

// PoolStats provides statistics about pool usage
type PoolStats struct {
	CandidatePoolHits uint64
	ResultPoolHits    uint64
	EmbeddingPoolHits uint64
	SlicePoolSizes    []int
}

// GetPoolStats returns current pool statistics (for monitoring)
func GetPoolStats() PoolStats {
	candidateSlicePool.mu.RLock()
	sizes := make([]int, 0, len(candidateSlicePool.pools))
	for size := range candidateSlicePool.pools {
		sizes = append(sizes, size)
	}
	candidateSlicePool.mu.RUnlock()

	return PoolStats{
		SlicePoolSizes: sizes,
	}
}
