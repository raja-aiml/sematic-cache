package reduction

import (
	"container/heap"
)

// MinHeap implements a min-heap for top-K selection
type MinHeap struct {
	items []HeapItem
	k     int
}

// HeapItem represents an item in the heap with its score
type HeapItem struct {
	Index      int
	Similarity float64
	Data       interface{}
}

// Implement heap.Interface
func (h *MinHeap) Len() int           { return len(h.items) }
func (h *MinHeap) Less(i, j int) bool { return h.items[i].Similarity < h.items[j].Similarity }
func (h *MinHeap) Swap(i, j int)      { h.items[i], h.items[j] = h.items[j], h.items[i] }

func (h *MinHeap) Push(x interface{}) {
	h.items = append(h.items, x.(HeapItem))
}

func (h *MinHeap) Pop() interface{} {
	old := h.items
	n := len(old)
	item := old[n-1]
	h.items = old[0 : n-1]
	return item
}

// NewMinHeap creates a new min-heap for top-K selection
func NewMinHeap(k int) *MinHeap {
	return &MinHeap{
		items: make([]HeapItem, 0, k+1),
		k:     k,
	}
}

// Add adds an item to the heap, maintaining top-K items
func (h *MinHeap) Add(item HeapItem) {
	if h.Len() < h.k {
		heap.Push(h, item)
	} else if item.Similarity > h.items[0].Similarity {
		// Replace the minimum with this item
		h.items[0] = item
		heap.Fix(h, 0)
	}
}

// GetTopK returns the top-K items sorted by similarity (descending)
func (h *MinHeap) GetTopK() []HeapItem {
	// Extract all items
	result := make([]HeapItem, h.Len())
	copy(result, h.items)
	
	// Sort by similarity descending
	// Since we have a min-heap, the items are already the top-K
	// but not necessarily in order. Sort them properly.
	for i := 0; i < len(result)-1; i++ {
		for j := i + 1; j < len(result); j++ {
			if result[i].Similarity < result[j].Similarity {
				result[i], result[j] = result[j], result[i]
			}
		}
	}
	
	return result
}

// TopKSelector provides efficient top-K selection using a min-heap
type TopKSelector struct {
	k int
}

// NewTopKSelector creates a new top-K selector
func NewTopKSelector(k int) *TopKSelector {
	return &TopKSelector{k: k}
}

// SelectTopK efficiently selects top-K items from candidates by similarity
func (s *TopKSelector) SelectTopK(candidates []scoredCandidate) []SearchCandidate {
	if len(candidates) <= s.k {
		// If we have fewer candidates than k, return all sorted
		sortBySimilarity(candidates)
		results := make([]SearchCandidate, len(candidates))
		for i, c := range candidates {
			results[i] = c.candidate
		}
		return results
	}

	// Use heap for efficient top-K selection
	minHeap := NewMinHeap(s.k)
	
	for i, candidate := range candidates {
		minHeap.Add(HeapItem{
			Index:      i,
			Similarity: candidate.similarity,
			Data:       candidate,
		})
	}
	
	// Get top-K items sorted by similarity
	topItems := minHeap.GetTopK()
	results := make([]SearchCandidate, len(topItems))
	
	for i, item := range topItems {
		results[i] = item.Data.(scoredCandidate).candidate
	}
	
	return results
}

// SelectTopKResults efficiently selects top-K results
func (s *TopKSelector) SelectTopKResults(results []scoredResult) []SearchResult {
	if len(results) <= s.k {
		// If we have fewer results than k, return all sorted
		sortResultsBySimilarity(results)
		output := make([]SearchResult, len(results))
		for i, r := range results {
			output[i] = r.result
		}
		return output
	}

	// Use heap for efficient top-K selection
	minHeap := NewMinHeap(s.k)
	
	for i, result := range results {
		minHeap.Add(HeapItem{
			Index:      i,
			Similarity: result.similarity,
			Data:       result,
		})
	}
	
	// Get top-K items sorted by similarity
	topItems := minHeap.GetTopK()
	output := make([]SearchResult, len(topItems))
	
	for i, item := range topItems {
		output[i] = item.Data.(scoredResult).result
	}
	
	return output
}

// BatchTopK efficiently processes multiple top-K queries in parallel
type BatchTopK struct {
	k       int
	workers int
}

// NewBatchTopK creates a new batch top-K processor
func NewBatchTopK(k, workers int) *BatchTopK {
	if workers <= 0 {
		workers = 4
	}
	return &BatchTopK{k: k, workers: workers}
}

// ProcessBatch processes multiple top-K queries efficiently
func (b *BatchTopK) ProcessBatch(queries []TopKQuery) [][]SearchResult {
	n := len(queries)
	results := make([][]SearchResult, n)
	
	// Process queries in parallel using worker pool
	chunkSize := (n + b.workers - 1) / b.workers
	done := make(chan int, b.workers)
	
	for w := 0; w < b.workers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > n {
			end = n
		}
		
		go func(start, end int) {
			selector := NewTopKSelector(b.k)
			for i := start; i < end; i++ {
				results[i] = selector.SelectTopKResults(queries[i].Candidates)
			}
			done <- 1
		}(start, end)
	}
	
	// Wait for all workers
	for w := 0; w < b.workers; w++ {
		<-done
	}
	
	return results
}

// TopKQuery represents a single top-K query
type TopKQuery struct {
	Candidates []scoredResult
}