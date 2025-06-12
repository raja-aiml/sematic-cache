package core

import "testing"

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
