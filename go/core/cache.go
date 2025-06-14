// Package core implements an in-memory cache for prompts and embeddings.

package core

import (
   "container/list"
   "fmt"
   "math"
   "math/rand"
   "sort"
   "sync"
   "sync/atomic"
   "time"
)
// Seed the random number generator for RR eviction
func init() {
   rand.Seed(time.Now().UnixNano())
}
// Eviction policy constants for in-memory cache
const (
   PolicyLRU  = "LRU"
   PolicyFIFO = "FIFO"
   PolicyLFU  = "LFU"
   PolicyRR   = "RR"
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
   // evictionPolicy determines the in-memory eviction strategy: "LRU", "FIFO", "LFU", or "RR"
   evictionPolicy string

	cacheEnable   func(string) bool
	embedFunc     EmbeddingFunc
	simFunc       func(a, b []float32) float64
   // minimum similarity threshold for brute-force searches
   minSimilarity     float64 // matches with similarity below this are ignored
   // adaptiveThreshold, if set, overrides minSimilarity per query using candidate sims
   adaptiveThreshold func([]float64) float64
   // ANNIndex for fast approximate searches; if set, GetByEmbedding/GetTopK use it first
   annIndex          ANNIndex
   // preProcess, if set, transforms the prompt before embedding or lookup
   preProcess        func(string) string
   // postProcess, if set, transforms the answer before returning
   postProcess       func(string) string
   // evaluators apply additional filtering or scoring to TopK results
   evaluators        []func([]QueryResult) []QueryResult
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
// WithEvictionPolicy sets the in-memory eviction strategy. Valid values are
// "LRU", "FIFO", "LFU", and "RR" (random replacement).
func WithEvictionPolicy(policy string) Option {
   return func(c *Cache) {
       switch policy {
       case PolicyLRU, PolicyFIFO, PolicyLFU, PolicyRR:
           c.evictionPolicy = policy
       default:
           panic(fmt.Sprintf("core: unknown eviction policy %q", policy))
       }
   }
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
// WithPreProcessor sets a function to transform prompts before caching or lookup.
func WithPreProcessor(fn func(string) string) Option {
   return func(c *Cache) { c.preProcess = fn }
}

// WithPostProcessor sets a function to transform answers before returning from lookup.
func WithPostProcessor(fn func(string) string) Option {
   return func(c *Cache) { c.postProcess = fn }
}

// WithEvaluationFunc adds a function to post-filter TopK results (after threshold and sort).
func WithEvaluationFunc(fn func([]QueryResult) []QueryResult) Option {
   return func(c *Cache) { c.evaluators = append(c.evaluators, fn) }
}

// NewCache creates a cache with the given capacity.
func NewCache(capacity int, opts ...Option) *Cache {
	// capacity must be positive
	if capacity <= 0 {
		panic("core: capacity must be > 0")
	}
   c := &Cache{
       entries:        make(map[string]*list.Element),
       lru:            list.New(),
       capacity:       capacity,
       evictionPolicy: PolicyLRU,
       cacheEnable:    func(string) bool { return true },
       simFunc:        cosine,
       minSimilarity:  -1.0,
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
   // apply pre-processing hook
   key := prompt
   if c.preProcess != nil {
       key = c.preProcess(prompt)
   }
   if c.cacheEnable != nil && !c.cacheEnable(key) {
       return
   }
   c.mu.Lock()
   defer c.mu.Unlock()

	now := nowUnix()
       if el, ok := c.entries[key]; ok {
           ent := el.Value.(*entry)
           ent.embedding = embedding
           ent.answer = answer
           ent.ModelName = modelName
           ent.ModelID = modelID
           // reset timestamp for TTL
           ent.timestamp = now
           ent.accessCount++
           ent.lastAccessed = now
           // update LRU position only for LRU policy
           if c.evictionPolicy == PolicyLRU {
               c.lru.MoveToFront(el)
           }
           return
       }

   ent := &entry{
       prompt:       key,
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
// Get returns the cached answer for a prompt and whether it was found.
// It applies any registered pre- and post-processing hooks.
func (c *Cache) Get(prompt string) (string, bool) {
   // apply pre-processing hook
   key := prompt
   if c.preProcess != nil {
       key = c.preProcess(prompt)
   }
   c.mu.Lock()
   defer c.mu.Unlock()
   if el, ok := c.entries[key]; ok {
       ent := el.Value.(*entry)
       // expire entry if needed
       if c.isExpired(ent) {
           c.lru.Remove(el)
           delete(c.entries, key)
           atomic.AddUint64(&c.missCount, 1)
           return "", false
       }
       // cache hit
       ent.accessCount++
       ent.lastAccessed = nowUnix()
       if c.evictionPolicy == PolicyLRU {
           c.lru.MoveToFront(el)
       }
       atomic.AddUint64(&c.hitCount, 1)
       // apply post-processing hook
       ans := ent.answer
       if c.postProcess != nil {
           ans = c.postProcess(ans)
       }
       return ans, true
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
   // Try ANN index for fast lookup when no adaptive threshold is set
   if c.annIndex != nil && c.adaptiveThreshold == nil {
       if keys, err := c.annIndex.Search(embed, 1); err == nil && len(keys) > 0 {
           // c.Get updates LRU and metrics
           return c.Get(keys[0])
       }
   }

   // Scan under read-lock to find the best candidate and collect expired keys.
   c.mu.RLock()
   var (
       bestKey          string
       bestSim          float64
       initBest         bool
       expiredKeys      []string
       sims             []float64
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
       if c.adaptiveThreshold != nil {
           sims = append(sims, sim)
       }
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
   // apply adaptive threshold if configured
   if c.adaptiveThreshold != nil {
       thr := c.adaptiveThreshold(sims)
       if bestSim < thr {
           atomic.AddUint64(&c.missCount, 1)
           return "", false
       }
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
   // update LRU position only for LRU policy
   if c.evictionPolicy == PolicyLRU {
       c.lru.MoveToFront(el)
   }
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
           // apply evaluation pipeline
           for _, eval := range c.evaluators {
               results = eval(results)
           }
           // apply post-processing hook
           if c.postProcess != nil {
               for i, r := range results {
                   r.Answer = c.postProcess(r.Answer)
                   results[i] = r
               }
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
   // apply evaluation pipeline
   for _, eval := range c.evaluators {
       results = eval(results)
   }
   // apply post-processing hook
   if c.postProcess != nil {
       for i, r := range results {
           r.Answer = c.postProcess(r.Answer)
           results[i] = r
       }
   }
   return results
}

// Flush clears all cached entries.
// Flush clears all cached entries (in-memory and ANN index).
func (c *Cache) Flush() {
   c.mu.Lock()
   defer c.mu.Unlock()
   // clear ANN index
   if c.annIndex != nil {
       for key := range c.entries {
           _ = c.annIndex.Remove(key)
       }
   }
   c.entries = make(map[string]*list.Element)
   c.lru.Init()
}

// insertEntry adds a new entry (assumes c.mu is held), evicting the oldest if over capacity.
func (c *Cache) insertEntry(ent *entry) {
	el := c.lru.PushFront(ent)
	c.entries[ent.prompt] = el
   if c.lru.Len() > c.capacity {
       // determine eviction key based on policy
       var evictKey string
       switch c.evictionPolicy {
       case PolicyLFU:
           // find entry with lowest accessCount
           minCount := int(^uint(0) >> 1) // max int
           for key, el0 := range c.entries {
               cnt := el0.Value.(*entry).accessCount
               if cnt < minCount {
                   minCount = cnt
                   evictKey = key
               }
           }
       case PolicyRR:
           // random replacement
           keys := make([]string, 0, len(c.entries))
           for key := range c.entries {
               keys = append(keys, key)
           }
           if len(keys) > 0 {
               idx := rand.Intn(len(keys))
               evictKey = keys[idx]
           }
       default:
           // LRU or FIFO: evict oldest in list (tail)
           if tail := c.lru.Back(); tail != nil {
               evictKey = tail.Value.(*entry).prompt
           }
       }
       // perform eviction
       if el0, ok := c.entries[evictKey]; ok {
           c.lru.Remove(el0)
           delete(c.entries, evictKey)
           if c.annIndex != nil {
               _ = c.annIndex.Remove(evictKey)
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
               // reset timestamp for TTL
               ent.timestamp = now
               ent.accessCount++
               ent.lastAccessed = now
               // update LRU only for LRU policy
               if c.evictionPolicy == PolicyLRU {
                   c.lru.MoveToFront(el)
               }
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
