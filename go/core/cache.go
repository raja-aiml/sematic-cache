// Package core implements an in-memory cache for prompts and embeddings.

package core

import (
	"container/list"
	"math"
	"sync"
)

// entry represents a cached item with its embedding.
type entry struct {
	prompt    string
	embedding []float32
	answer    string
}

// Cache provides a concurrent LRU cache storing embeddings and answers.
type Cache struct {
	mu       sync.Mutex
	entries  map[string]*list.Element
	lru      *list.List
	capacity int
}

// NewCache creates a cache with the given capacity.
func NewCache(capacity int) *Cache {
	return &Cache{
		entries:  make(map[string]*list.Element),
		lru:      list.New(),
		capacity: capacity,
	}
}

// Set stores an answer and embedding for the given prompt.
func (c *Cache) Set(prompt string, embedding []float32, answer string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if el, ok := c.entries[prompt]; ok {
		ent := el.Value.(*entry)
		ent.embedding = embedding
		ent.answer = answer
		c.lru.MoveToFront(el)
		return
	}

	ent := &entry{prompt: prompt, embedding: embedding, answer: answer}
	el := c.lru.PushFront(ent)
	c.entries[prompt] = el

	if c.lru.Len() > c.capacity {
		tail := c.lru.Back()
		if tail != nil {
			c.lru.Remove(tail)
			delete(c.entries, tail.Value.(*entry).prompt)
		}
	}
}

// Get returns the cached answer for a prompt and whether it was found.
func (c *Cache) Get(prompt string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.entries[prompt]; ok {
		c.lru.MoveToFront(el)
		return el.Value.(*entry).answer, true
	}
	return "", false
}

// GetByEmbedding returns the answer whose embedding is most similar to the query.
func (c *Cache) GetByEmbedding(embed []float32) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	var (
		best     *list.Element
		bestSim  float64
		initBest bool
	)
	for e := c.lru.Front(); e != nil; e = e.Next() {
		ent := e.Value.(*entry)
		if len(ent.embedding) != len(embed) || len(embed) == 0 {
			continue
		}
		sim := cosine(ent.embedding, embed)
		if !initBest || sim > bestSim {
			best = e
			bestSim = sim
			initBest = true
		}
	}
	if best != nil {
		c.lru.MoveToFront(best)
		return best.Value.(*entry).answer, true
	}
	return "", false
}

// Flush clears all cached entries.
func (c *Cache) Flush() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries = make(map[string]*list.Element)
	c.lru.Init()
}

// ImportData loads multiple prompt/embedding/answer triples into the cache.
func (c *Cache) ImportData(prompts []string, embeddings [][]float32, answers []string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for i, p := range prompts {
		var e []float32
		if i < len(embeddings) {
			e = embeddings[i]
		}
		var a string
		if i < len(answers) {
			a = answers[i]
		}
		if el, ok := c.entries[p]; ok {
			ent := el.Value.(*entry)
			ent.embedding = e
			ent.answer = a
			c.lru.MoveToFront(el)
			continue
		}
		ent := &entry{prompt: p, embedding: e, answer: a}
		el := c.lru.PushFront(ent)
		c.entries[p] = el
		if c.lru.Len() > c.capacity {
			tail := c.lru.Back()
			if tail != nil {
				c.lru.Remove(tail)
				delete(c.entries, tail.Value.(*entry).prompt)
			}
		}
	}
}

func cosine(a, b []float32) float64 {
	var dot, aa, bb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	if aa == 0 || bb == 0 {
		return 0
	}
	return dot / (math.Sqrt(aa) * math.Sqrt(bb))
}
