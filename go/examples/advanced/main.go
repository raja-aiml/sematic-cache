// Package main demonstrates advanced usage of the semantic cache:
// inner-product similarity, a naive ANNIndex plugin, and adaptive thresholding.
package main

import (
	"fmt"
	"sort"

	"github.com/raja-aiml/sematic-cache/go/core"
)

// NaiveANNIndex is a simple brute-force ANNIndex implementation.
type NaiveANNIndex struct {
	store map[string][]float32
}

// NewNaiveANNIndex creates an empty NaiveANNIndex.
func NewNaiveANNIndex() *NaiveANNIndex {
	return &NaiveANNIndex{store: make(map[string][]float32)}
}

// Add inserts a key and vector into the index.
func (n *NaiveANNIndex) Add(key string, v []float32) error {
	n.store[key] = v
	return nil
}

// Remove deletes a key from the index.
func (n *NaiveANNIndex) Remove(key string) error {
	delete(n.store, key)
	return nil
}

// Search returns up to k keys whose vectors have highest inner-product with the query.
func (n *NaiveANNIndex) Search(query []float32, k int) ([]string, error) {
	type res struct {
		key string
		sim float64
	}
	var results []res
	for key, v := range n.store {
		if len(v) != len(query) {
			continue
		}
		sim := core.InnerProduct(v, query)
		results = append(results, res{key, sim})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].sim > results[j].sim
	})
	var keys []string
	for i := 0; i < len(results) && i < k; i++ {
		keys = append(keys, results[i].key)
	}
	return keys, nil
}

func main() {
	// Prepare a naive ANN index and a cache using inner-product similarity.
	ann := NewNaiveANNIndex()
	cache := core.NewCache(10,
		core.WithInnerProduct(),
		core.WithANNIndex(ann),
		core.WithMinSimilarity(0.0),
	)

	// Sample data: prompt -> (vector, answer)
	data := map[string]struct {
		vec []float32
		ans string
	}{
		"apple":  {[]float32{1, 0}, "A fruit"},
		"banana": {[]float32{0, 1}, "Yellow fruit"},
		"cherry": {[]float32{0.9, 0.1}, "Small red fruit"},
	}
	for prompt, d := range data {
		cache.Set(prompt, d.vec, d.ans)
	}

	// Query the cache by embedding vector.
	query := []float32{0.95, 0.05}
	ans, found := cache.GetByEmbedding(query)
	fmt.Printf("GetByEmbedding(%.2v) -> %q (found=%v)\n", query, ans, found)

	// Retrieve top-2 matches.
	top2 := cache.GetTopKByEmbedding(query, 2)
	fmt.Println("Top 2 matches:")
	for _, r := range top2 {
		fmt.Printf("  %s (sim=%.4f): %s\n", r.Prompt, r.Similarity, r.Answer)
	}

	// Adaptive threshold example: keep only sims >= mean(sim)
	meanThreshold := func(sims []float64) float64 {
		var sum float64
		for _, v := range sims {
			sum += v
		}
		return sum / float64(len(sims))
	}
	cache2 := core.NewCache(10,
		core.WithInnerProduct(),
		core.WithANNIndex(ann),
		core.WithAdaptiveThreshold(meanThreshold),
	)
	for prompt, d := range data {
		cache2.Set(prompt, d.vec, d.ans)
	}
	fmt.Println("Adaptive threshold top-K:")
	for _, r := range cache2.GetTopKByEmbedding(query, 3) {
		fmt.Printf("  %s (sim=%.4f): %s\n", r.Prompt, r.Similarity, r.Answer)
	}
}
