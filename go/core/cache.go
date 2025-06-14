// Package core implements an in-memory cache for prompts and embeddings.

package core

import (
   "container/list"
   "fmt"
   "math"
   "sort"
   "sync"
   "sync/atomic"
   "time"
)
// ANNIndex defines an interface for approximate nearest-neighbor search.
// Users can plug in HNSW, LSH, or other ANN implementations.
type ANNIndex interface {
   // Add inserts an item with the given key and vector into the index.
   Add(key string, vector []float32) error
   // Remove deletes an item by key from the index.
   Remove(key string) error
   // Search returns up to k keys whose vectors are nearest to the query vector.
   Search(vector []float32, k int) ([]string, error)
}

// Similarity functions
// InnerProduct returns the dot-product similarity of a and b.
func InnerProduct(a, b []float32) float64 {
   var sum float64
   for i := range a {
       sum += float64(a[i]) * float64(b[i])
   }
   return sum
}

// L2Similarity returns a similarity derived from Euclidean distance: 1/(1+distance).
func L2Similarity(a, b []float32) float64 {
   var dist2 float64
   for i := range a {
       d := float64(a[i] - b[i])
       dist2 += d * d
   }
   return 1.0 / (1.0 + math.Sqrt(dist2))
}

// entry represents a cached item with its embedding and metadata.
type entry struct {
	prompt    string
	embedding []float32
	answer    string
	// model metadata
	ModelName string // e.g. "gpt-3.5-turbo"
	ModelID   string // e.g. deployment or version identifier
	// metadata
	accessCount  int   // number of times this entry was retrieved
	lastAccessed int64 // UnixNano timestamp of last access
	timestamp    int64 // UnixNano timestamp when stored
	// future: contextChain, expertID, etc.
}

// Stats returns the number of cache hits, misses, and the hit rate.
func (c *Cache) Stats() (hits, misses uint64, hitRate float64) {
	hits = atomic.LoadUint64(&c.hitCount)
	misses = atomic.LoadUint64(&c.missCount)
	total := hits + misses
	if total > 0 {
		hitRate = float64(hits) / float64(total)
	}
	return
}

// SetBatch inserts multiple entries into the cache (no model metadata).
func (c *Cache) SetBatch(prompts []string, embeddings [][]float32, answers []string) {
	for i, prompt := range prompts {
		var emb []float32
		if i < len(embeddings) {
			emb = embeddings[i]
		}
		var ans string
		if i < len(answers) {
			ans = answers[i]
		}
		c.Set(prompt, emb, ans)
	}
}

// GetBatch retrieves multiple prompts from the cache, returning a map of found answers.
func (c *Cache) GetBatch(prompts []string) map[string]string {
	results := make(map[string]string, len(prompts))
	for _, prompt := range prompts {
		if ans, ok := c.Get(prompt); ok {
			results[prompt] = ans
		}
	}
	return results
}

// nowUnix returns the current Unix timestamp (seconds).
// nowUnix returns the current Unix timestamp in nanoseconds.
func nowUnix() int64 {
	return time.Now().UnixNano()
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
	mu       sync.RWMutex
	entries  map[string]*list.Element
	lru      *list.List
	capacity int

	cacheEnable   func(string) bool
	embedFunc     EmbeddingFunc
	simFunc       func(a, b []float32) float64
   // minimum similarity threshold for brute-force searches
   minSimilarity     float64 // matches with similarity below this are ignored
   // adaptiveThreshold, if set, overrides minSimilarity per query using candidate sims
   adaptiveThreshold func([]float64) float64
   // ANNIndex for fast approximate searches; if set, GetByEmbedding/GetTopK use it first
   annIndex          ANNIndex
   // TTL for entries; if >0, entries older than now-TTL are expired
   ttl time.Duration
   // metrics
   hitCount  uint64 // number of cache hits
   missCount uint64 // number of cache misses
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

// WithTTL sets a time-to-live for cache entries. Entries older than now-TTL
// will be considered expired and evicted on access.
func WithTTL(d time.Duration) Option {
	return func(c *Cache) { c.ttl = d }
}

// WithInnerProduct sets the similarity metric to raw dot-product.
func WithInnerProduct() Option {
   return func(c *Cache) { c.simFunc = InnerProduct }
}

// WithL2Similarity sets the similarity metric to an L2-based score (1/(1+distance)).
func WithL2Similarity() Option {
   return func(c *Cache) { c.simFunc = L2Similarity }
}

// WithAdaptiveThreshold sets a function that computes a dynamic threshold
// from a slice of candidate similarities. If provided, this overrides minSimilarity per query.
func WithAdaptiveThreshold(fn func([]float64) float64) Option {
   return func(c *Cache) { c.adaptiveThreshold = fn }
}

// WithANNIndex enables an approximate nearest-neighbor index for fast searches.
// The ANNIndex should implement Add, Remove, and Search methods.
func WithANNIndex(idx ANNIndex) Option {
   return func(c *Cache) { c.annIndex = idx }
}

// NewCache creates a cache with the given capacity.
func NewCache(capacity int, opts ...Option) *Cache {
	// capacity must be positive
	if capacity <= 0 {
		panic("core: capacity must be > 0")
	}
	c := &Cache{
		entries:       make(map[string]*list.Element),
		lru:           list.New(),
		capacity:      capacity,
		cacheEnable:   func(string) bool { return true },
		simFunc:       cosine,
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
   // add to ANN index if configured
   if c.annIndex != nil {
       _ = c.annIndex.Add(prompt, embedding)
   }
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
		// expire entry if needed
		if c.isExpired(ent) {
			// evict expired entry
			c.lru.Remove(el)
			delete(c.entries, prompt)
			atomic.AddUint64(&c.missCount, 1)
			return "", false
		}
		// cache hit
		ent.accessCount++
		ent.lastAccessed = nowUnix()
		c.lru.MoveToFront(el)
		atomic.AddUint64(&c.hitCount, 1)
		return ent.answer, true
	}
	atomic.AddUint64(&c.missCount, 1)
	return "", false
}

// GetByEmbedding returns the answer whose embedding is most similar to the query.
func (c *Cache) GetByEmbedding(embed []float32) (string, bool) {
   // Return early on empty embedding.
   if len(embed) == 0 {
       atomic.AddUint64(&c.missCount, 1)
       return "", false
   }
   // Try ANN index if configured for fast lookup
   if c.annIndex != nil {
       if keys, err := c.annIndex.Search(embed, 1); err == nil && len(keys) > 0 {
           // c.Get updates LRU and metrics
           return c.Get(keys[0])
       }
   }

   // Scan under read-lock to find the best candidate and collect expired keys.
   c.mu.RLock()
   var (
       bestKey     string
       bestSim     float64
       initBest    bool
       expiredKeys []string
   )
   for e := c.lru.Front(); e != nil; e = e.Next() {
       ent := e.Value.(*entry)
       if len(ent.embedding) != len(embed) {
           continue
       }
       if c.isExpired(ent) {
           expiredKeys = append(expiredKeys, ent.prompt)
           continue
       }
       sim := c.simFunc(ent.embedding, embed)
       if sim < c.minSimilarity {
           continue
       }
       if !initBest || sim > bestSim {
           bestKey = ent.prompt
           bestSim = sim
           initBest = true
       }
   }
   c.mu.RUnlock()

   // Evict expired entries under write lock.
   if len(expiredKeys) > 0 {
       c.mu.Lock()
       for _, key := range expiredKeys {
           if el, ok := c.entries[key]; ok {
               c.lru.Remove(el)
               delete(c.entries, key)
           }
       }
       c.mu.Unlock()
   }

   // If no match, record miss.
   if !initBest {
       atomic.AddUint64(&c.missCount, 1)
       return "", false
   }

   // Update metadata on the chosen entry under write lock.
   c.mu.Lock()
   el, ok := c.entries[bestKey]
   if !ok {
       atomic.AddUint64(&c.missCount, 1)
       c.mu.Unlock()
       return "", false
   }
   ent := el.Value.(*entry)
   if c.isExpired(ent) {
       c.lru.Remove(el)
       delete(c.entries, bestKey)
       atomic.AddUint64(&c.missCount, 1)
       c.mu.Unlock()
       return "", false
   }
   ent.accessCount++
   ent.lastAccessed = nowUnix()
   c.lru.MoveToFront(el)
   c.mu.Unlock()
   atomic.AddUint64(&c.hitCount, 1)
   return ent.answer, true
}

// GetTopKByEmbedding returns up to k answers whose embeddings are most similar to the query.
// It supports ANN-based search and adaptive thresholding.
func (c *Cache) GetTopKByEmbedding(embed []float32, k int) []QueryResult {
   if len(embed) == 0 || k <= 0 {
       return nil
   }
   // Try ANN index if configured
   if c.annIndex != nil {
       if keys, err := c.annIndex.Search(embed, k); err == nil && len(keys) > 0 {
           var results []QueryResult
           c.mu.RLock()
           for _, key := range keys {
               if el, ok := c.entries[key]; ok {
                   ent := el.Value.(*entry)
                   if c.isExpired(ent) {
                       continue
                   }
                   sim := c.simFunc(ent.embedding, embed)
                   if sim < c.minSimilarity {
                       continue
                   }
                   results = append(results, QueryResult{
                       Prompt:     ent.prompt,
                       Answer:     ent.answer,
                       Similarity: sim,
                       ModelName:  ent.ModelName,
                       ModelID:    ent.ModelID,
                   })
               }
           }
           c.mu.RUnlock()
           sort.Slice(results, func(i, j int) bool {
               return results[i].Similarity > results[j].Similarity
           })
           if c.adaptiveThreshold != nil && len(results) > 0 {
               sims := make([]float64, len(results))
               for i, r := range results {
                   sims[i] = r.Similarity
               }
               thr := c.adaptiveThreshold(sims)
               filtered := results[:0]
               for _, r := range results {
                   if r.Similarity >= thr {
                       filtered = append(filtered, r)
                   }
               }
               results = filtered
           }
           if len(results) > k {
               results = results[:k]
           }
           return results
       }
   }
   // Fallback brute-force
   c.mu.RLock()
   var results []QueryResult
   for key, el := range c.entries {
       ent := el.Value.(*entry)
       if c.isExpired(ent) {
           continue
       }
       if len(ent.embedding) != len(embed) {
           continue
       }
       sim := c.simFunc(ent.embedding, embed)
       if sim < c.minSimilarity {
           continue
       }
       results = append(results, QueryResult{
           Prompt:     key,
           Answer:     ent.answer,
           Similarity: sim,
           ModelName:  ent.ModelName,
           ModelID:    ent.ModelID,
       })
   }
   c.mu.RUnlock()
   sort.Slice(results, func(i, j int) bool {
       return results[i].Similarity > results[j].Similarity
   })
   if c.adaptiveThreshold != nil && len(results) > 0 {
       sims := make([]float64, len(results))
       for i, r := range results {
           sims[i] = r.Similarity
       }
       thr := c.adaptiveThreshold(sims)
       filtered := results[:0]
       for _, r := range results {
           if r.Similarity >= thr {
               filtered = append(filtered, r)
           }
       }
       results = filtered
   }
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
			oldKey := tail.Value.(*entry).prompt
			c.lru.Remove(tail)
			delete(c.entries, oldKey)
			// remove from ANN index if present
			if c.annIndex != nil {
				_ = c.annIndex.Remove(oldKey)
			}
		}
	}
}

// isExpired returns true if the entry is older than the cache TTL.
func (c *Cache) isExpired(ent *entry) bool {
	if c.ttl <= 0 {
		return false
	}
	return time.Duration(nowUnix()-ent.timestamp) > c.ttl
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
