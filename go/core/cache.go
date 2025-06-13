// Package core implements an in-memory cache for prompts and embeddings.

package core

import (
   "container/list"
   "fmt"
   "math"
   "sort"
   "sync"
   "time"
)

// entry represents a cached item with its embedding and metadata.
type entry struct {
   prompt             string
   embedding          []float32
   answer             string
   // model metadata
   ModelName          string  // e.g. "gpt-3.5-turbo"
   ModelID            string  // e.g. deployment or version identifier
   // metadata
   accessCount        int       // number of times this entry was retrieved
   lastAccessed       int64     // Unix timestamp of last access
   timestamp          int64     // Unix timestamp when stored
   // future: contextChain, expertID, etc.
}

// nowUnix returns the current Unix timestamp (seconds).
func nowUnix() int64 {
   return time.Now().Unix()
}

// QueryResult holds a single match from a similarity search, including model metadata.
type QueryResult struct {
   Prompt     string
   Answer     string
   Similarity float64
   ModelName  string
   ModelID    string
}

// Cache provides a concurrent LRU cache storing embeddings and answers.
// It supports cosine similarity search with optional threshold and top-K queries.
type Cache struct {
   mu             sync.RWMutex
	entries  map[string]*list.Element
	lru      *list.List
	capacity int

   cacheEnable    func(string) bool
   embedFunc      EmbeddingFunc
   simFunc        func(a, b []float32) float64
   minSimilarity  float64  // minimum similarity threshold for search (default: -1)
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
// WithMinSimilarity sets the minimum similarity threshold for GetByEmbedding and TopK searches.
// Matches with similarity below this value are ignored.
func WithMinSimilarity(th float64) Option {
   return func(c *Cache) { c.minSimilarity = th }
}

// NewCache creates a cache with the given capacity.
func NewCache(capacity int, opts ...Option) *Cache {
	// capacity must be positive
	if capacity <= 0 {
		panic("core: capacity must be > 0")
	}
	c := &Cache{
       entries:      make(map[string]*list.Element),
		lru:         list.New(),
		capacity:    capacity,
		cacheEnable: func(string) bool { return true },
		simFunc:     cosine,
       minSimilarity: -1.0,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Set stores an answer and embedding for the given prompt, without model metadata.
// It is equivalent to SetWithModel(prompt, embedding, answer, "", "").
func (c *Cache) Set(prompt string, embedding []float32, answer string) {
   c.SetWithModel(prompt, embedding, answer, "", "")
}

// SetWithModel stores an answer and embedding for the given prompt, including model metadata.
// modelName is the model's name (e.g. "gpt-3.5-turbo"); modelID is the specific deployment or version identifier.
func (c *Cache) SetWithModel(prompt string, embedding []float32, answer, modelName, modelID string) {
   if c.cacheEnable != nil && !c.cacheEnable(prompt) {
       return
   }
   c.mu.Lock()
   defer c.mu.Unlock()

   now := nowUnix()
   if el, ok := c.entries[prompt]; ok {
       ent := el.Value.(*entry)
       ent.embedding = embedding
       ent.answer = answer
       ent.ModelName = modelName
       ent.ModelID = modelID
       ent.timestamp = now
       ent.accessCount++
       ent.lastAccessed = now
       c.lru.MoveToFront(el)
       return
   }

   ent := &entry{
       prompt:       prompt,
       embedding:    embedding,
       answer:       answer,
       ModelName:    modelName,
       ModelID:      modelID,
       timestamp:    now,
       lastAccessed: now,
       accessCount:  1,
   }
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
       ent := el.Value.(*entry)
       // update metadata
       ent.accessCount++
       ent.lastAccessed = nowUnix()
       c.lru.MoveToFront(el)
       return ent.answer, true
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
       // apply threshold filter
       if sim < c.minSimilarity {
           continue
       }
       if !initBest || sim > bestSim {
           best = e
           bestSim = sim
           initBest = true
       }
   }
   if best != nil {
       ent := best.Value.(*entry)
       // update metadata
       ent.accessCount++
       ent.lastAccessed = nowUnix()
       c.lru.MoveToFront(best)
       return ent.answer, true
   }
	return "", false
}
// GetTopKByEmbedding returns up to k answers whose embeddings are most similar to the query.
// Matches with similarity below the configured threshold are ignored.
func (c *Cache) GetTopKByEmbedding(embed []float32, k int) []QueryResult {
   c.mu.RLock()
   defer c.mu.RUnlock()
   var results []QueryResult
   // iterate through entries
   for prompt, el := range c.entries {
       ent := el.Value.(*entry)
       if len(ent.embedding) != len(embed) || len(embed) == 0 {
           continue
       }
       sim := c.simFunc(ent.embedding, embed)
       if sim < c.minSimilarity {
           continue
       }
       results = append(results, QueryResult{
           Prompt:     prompt,
           Answer:     ent.answer,
           Similarity: sim,
           ModelName:  ent.ModelName,
           ModelID:    ent.ModelID,
       })
   }
   // sort descending by similarity
   sort.Slice(results, func(i, j int) bool {
       return results[i].Similarity > results[j].Similarity
   })
   if len(results) > k {
       results = results[:k]
   }
   return results
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

// GetModelInfo returns the modelName and modelID for a given prompt, and whether the prompt exists in the cache.
// Both strings will be empty if found == false.
func (c *Cache) GetModelInfo(prompt string) (modelName, modelID string, found bool) {
   c.mu.RLock()
   defer c.mu.RUnlock()
   if el, ok := c.entries[prompt]; ok {
       ent := el.Value.(*entry)
       return ent.ModelName, ent.ModelID, true
   }
   return "", "", false
}

// ImportData loads multiple prompt/embedding/answer triples into the cache.
func (c *Cache) ImportData(prompts []string, embeddings [][]float32, answers []string) {
	c.mu.Lock()
	defer c.mu.Unlock()
   for i, p := range prompts {
       now := nowUnix()
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
           ent.accessCount++
           ent.lastAccessed = now
           c.lru.MoveToFront(el)
           continue
       }
       ent := &entry{
           prompt:       p,
           embedding:    e,
           answer:       a,
           timestamp:    now,
           lastAccessed: now,
           accessCount:  1,
       }
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
