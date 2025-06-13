// Package core implements an in-memory cache for prompts and embeddings.

package core

import (
	"container/list"
	"fmt"
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

	cacheEnable func(string) bool
	embedFunc   EmbeddingFunc
	simFunc     func(a, b []float32) float64
}

// EmbeddingFunc converts a prompt into an embedding vector.
// It may return an error if embedding fails.
type EmbeddingFunc func(string) ([]float32, error)

// Option configures a Cache.
type Option func(*Cache)

// WithEmbeddingFunc sets a function used to generate embeddings when calling
// SetPrompt. If not provided, SetPrompt will return an error.
func WithEmbeddingFunc(fn EmbeddingFunc) Option { return func(c *Cache) { c.embedFunc = fn } }

// WithCacheEnable sets a function that determines whether a prompt should be
// cached. If nil, all prompts are cached.
func WithCacheEnable(fn func(string) bool) Option { return func(c *Cache) { c.cacheEnable = fn } }

// WithSimilarityFunc overrides the vector similarity metric used by GetByEmbedding.
// By default cosine similarity is used.
func WithSimilarityFunc(fn func(a, b []float32) float64) Option {
	return func(c *Cache) { c.simFunc = fn }
}

// NewCache creates a cache with the given capacity.
func NewCache(capacity int, opts ...Option) *Cache {
	// capacity must be positive
	if capacity <= 0 {
		panic("core: capacity must be > 0")
	}
	c := &Cache{
		entries:     make(map[string]*list.Element),
		lru:         list.New(),
		capacity:    capacity,
		cacheEnable: func(string) bool { return true },
		simFunc:     cosine,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Set stores an answer and embedding for the given prompt.
func (c *Cache) Set(prompt string, embedding []float32, answer string) {
	if c.cacheEnable != nil && !c.cacheEnable(prompt) {
		return
	}
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
	c.insertEntry(ent)
}

// SetPrompt generates an embedding for the prompt using the configured
// EmbeddingFunc and stores the answer in the cache.
// It returns an error if no EmbeddingFunc is configured or embedding fails.
func (c *Cache) SetPrompt(prompt, answer string) error {
	if c.embedFunc == nil {
		return fmt.Errorf("no EmbeddingFunc configured")
	}
	emb, err := c.embedFunc(prompt)
	if err != nil {
		return err
	}
	c.Set(prompt, emb, answer)
	return nil
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
		sim := c.simFunc(ent.embedding, embed)
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

// insertEntry adds a new entry (assumes c.mu is held), evicting the oldest if over capacity.
func (c *Cache) insertEntry(ent *entry) {
	el := c.lru.PushFront(ent)
	c.entries[ent.prompt] = el
	if c.lru.Len() > c.capacity {
		tail := c.lru.Back()
		if tail != nil {
			c.lru.Remove(tail)
			delete(c.entries, tail.Value.(*entry).prompt)
		}
	}
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
		c.insertEntry(ent)
	}
}

// cosine returns the cosine similarity between two vectors (range [-1,1]).
// If either vector has zero magnitude, cosine returns 0.
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
