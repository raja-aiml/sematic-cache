package core

import (
	"container/list"
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
	return dot / (sqrt(aa) * sqrt(bb))
}

func sqrt(v float64) float64 {
	// simple Newton method
	x := v
	for i := 0; i < 10; i++ {
		x = 0.5 * (x + v/x)
	}
	return x
}
