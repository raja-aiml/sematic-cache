package core

import (
   "fmt"
   "testing"
   "time"
)

func TestCacheSetGet(t *testing.T) {
	c := NewCache(2)
	c.Set("p1", []float32{1, 0}, "a1")
	c.Set("p2", []float32{0, 1}, "a2")

	if val, ok := c.Get("p1"); !ok || val != "a1" {
		t.Fatalf("expected a1, got %v %v", val, ok)
	}

	// trigger eviction
	c.Set("p3", []float32{1, 1}, "a3")
	if _, ok := c.Get("p2"); ok {
		t.Fatalf("expected p2 evicted")
	}
}

// Test entry expiration via TTL.
func TestTTLExpiration(t *testing.T) {
   // TTL of 50ms
   c := NewCache(2, WithTTL(50*time.Millisecond))
   c.Set("p1", []float32{1}, "a1")
   // immediately retrievable
   if v, ok := c.Get("p1"); !ok || v != "a1" {
       t.Fatalf("expected initial hit, got %v %v", v, ok)
   }
   // wait for expiration
   time.Sleep(60 * time.Millisecond)
   if _, ok := c.Get("p1"); ok {
       t.Fatalf("expected entry expired after TTL")
   }
}

// Test cache metrics: hits, misses, hit rate.
func TestStats(t *testing.T) {
   c := NewCache(2)
   // misses
   c.Get("x")
   // hits
   c.Set("p", []float32{1}, "a")
   c.Get("p")
   hits, misses, rate := c.Stats()
   if hits != 1 || misses != 1 {
       t.Errorf("expected hits=1 misses=1, got %d %d", hits, misses)
   }
   if rate != 0.5 {
       t.Errorf("expected hit rate 0.5, got %v", rate)
   }
}

// Test batch set and get operations.
func TestBatchSetGet(t *testing.T) {
   c := NewCache(5)
   prompts := []string{"a", "b", "c"}
   embeddings := [][]float32{{1}, {2}, {3}}
   answers := []string{"A", "B", "C"}
   c.SetBatch(prompts, embeddings, answers)
   res := c.GetBatch([]string{"a", "c", "d"})
   if len(res) != 2 {
       t.Fatalf("expected 2 entries, got %d", len(res))
   }
   if res["a"] != "A" || res["c"] != "C" {
       t.Errorf("unexpected batch get results: %v", res)
   }
}

func TestCacheGetByEmbedding(t *testing.T) {
	c := NewCache(2)
	c.Set("p1", []float32{1, 0}, "a1")
	c.Set("p2", []float32{0, 1}, "a2")

	ans, ok := c.GetByEmbedding([]float32{0, 0.9})
	if !ok || ans != "a2" {
		t.Fatalf("expected a2, got %v %v", ans, ok)
	}
}

func TestCacheFlushAndImport(t *testing.T) {
	c := NewCache(3)
	c.Set("p1", []float32{1, 0}, "a1")
	c.Flush()
	if _, ok := c.Get("p1"); ok {
		t.Fatalf("expected empty cache")
	}

	prompts := []string{"p2", "p3"}
	embeddings := [][]float32{{0, 1}, {1, 1}}
	answers := []string{"a2", "a3"}
	c.ImportData(prompts, embeddings, answers)
	if val, ok := c.Get("p2"); !ok || val != "a2" {
		t.Fatalf("import failed")
	}
}

func TestCacheSetPrompt(t *testing.T) {
	embFn := func(p string) ([]float32, error) {
		if p == "p" {
			return []float32{1, 0}, nil
		}
		return nil, fmt.Errorf("unknown prompt")
	}
	c := NewCache(2, WithEmbeddingFunc(embFn))
	if err := c.SetPrompt("p", "a"); err != nil {
		t.Fatalf("SetPrompt error: %v", err)
	}
	if ans, ok := c.Get("p"); !ok || ans != "a" {
		t.Fatalf("expected cached answer, got %v %v", ans, ok)
	}
}

// Test that GetByEmbedding respects the min similarity threshold.
func TestGetByEmbeddingThreshold(t *testing.T) {
   // insert two orthogonal unit vectors
   c := NewCache(2, WithMinSimilarity(0.8))
   c.Set("p1", []float32{1, 0}, "a1")
   c.Set("p2", []float32{0, 1}, "a2")
   // query exactly matches p1 -> sim=1
   if ans, ok := c.GetByEmbedding([]float32{1, 0}); !ok || ans != "a1" {
       t.Fatalf("expected a1 above threshold, got %v %v", ans, ok)
   }
   // query diagonal [1,1] has sim~0.707<0.8 -> no match
   if ans, ok := c.GetByEmbedding([]float32{1, 1}); ok {
       t.Fatalf("expected no match when all sims < threshold, got %v", ans)
   }
}

// Test top-K similarity search.
func TestGetTopKByEmbedding(t *testing.T) {
   c := NewCache(3)
   c.Set("p1", []float32{1, 0}, "a1")
   c.Set("p2", []float32{0, 1}, "a2")
   c.Set("p3", []float32{1, 1}, "a3")
   // query [1,1] should rank p3 first, then p1 and p2 (equal)
   res := c.GetTopKByEmbedding([]float32{1, 1}, 2)
   if len(res) != 2 {
       t.Fatalf("expected 2 results, got %d", len(res))
   }
   if res[0].Answer != "a3" {
       t.Errorf("expected first a3, got %v", res[0].Answer)
   }
   // second could be a1 or a2; both have cos sim ~0.707
   if res[1].Answer != "a1" && res[1].Answer != "a2" {
       t.Errorf("expected second a1 or a2, got %v", res[1].Answer)
   }
}
