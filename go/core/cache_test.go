package core

import (
	"fmt"
	"testing"
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
