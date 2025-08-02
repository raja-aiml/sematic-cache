package cache

import (
	"math/rand"
	"testing"
)

// TestLRUPolicy ensures that LRU eviction removes the least-recently-used item.
func TestLRUPolicy(t *testing.T) {
	opt, err := WithEvictionPolicy(PolicyLRU)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	cache, err := NewCache(2, opt)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	cache.Set("a", []float32{1}, "A")
	cache.Set("b", []float32{2}, "B")
	// Access "a" to make it most-recent
	if val, ok := cache.Get("a"); !ok || val != "A" {
		t.Fatalf("expected hit for 'a'")
	}
	// Insert "c", should evict "b"
	cache.Set("c", []float32{3}, "C")
	if _, ok := cache.Get("b"); ok {
		t.Errorf("LRU: expected 'b' to be evicted")
	}
	// "a" and "c" should remain
	if _, ok := cache.Get("a"); !ok {
		t.Errorf("LRU: expected 'a' to remain")
	}
	if _, ok := cache.Get("c"); !ok {
		t.Errorf("LRU: expected 'c' to be present")
	}
}

// TestFIFOPolicy ensures that FIFO eviction removes the entry inserted earliest.
func TestFIFOPolicy(t *testing.T) {
	opt, err := WithEvictionPolicy(PolicyFIFO)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	cache, err := NewCache(2, opt)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	cache.Set("a", []float32{1}, "A")
	cache.Set("b", []float32{2}, "B")
	// Access "a" should not affect FIFO order
	if val, ok := cache.Get("a"); !ok || val != "A" {
		t.Fatalf("expected hit for 'a'")
	}
	// Insert "c", should evict "a" (the first inserted)
	cache.Set("c", []float32{3}, "C")
	if _, ok := cache.Get("a"); ok {
		t.Errorf("FIFO: expected 'a' to be evicted")
	}
	if _, ok := cache.Get("b"); !ok {
		t.Errorf("FIFO: expected 'b' to remain")
	}
	if _, ok := cache.Get("c"); !ok {
		t.Errorf("FIFO: expected 'c' to be present")
	}
}

// TestLFUPolicy ensures that LFU eviction removes the least-frequently-used item.
func TestLFUPolicy(t *testing.T) {
	opt, err := WithEvictionPolicy(PolicyLFU)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	cache, err := NewCache(2, opt)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	cache.Set("a", []float32{1}, "A")
	cache.Set("b", []float32{2}, "B")
	// Access frequencies: a:2 (1 initial +1), b:1
	if _, ok := cache.Get("a"); !ok {
		t.Fatalf("expected hit for 'a'")
	}
	// Insert "c", should evict "b" (LFU)
	cache.Set("c", []float32{3}, "C")
	if _, ok := cache.Get("b"); ok {
		t.Errorf("LFU: expected 'b' to be evicted")
	}
	if _, ok := cache.Get("a"); !ok {
		t.Errorf("LFU: expected 'a' to remain")
	}
	if _, ok := cache.Get("c"); !ok {
		t.Errorf("LFU: expected 'c' to be present")
	}
}

// TestRRPolicy ensures that RR eviction removes a random entry (non-deterministic behavior).
func TestRRPolicy(t *testing.T) {
	// Seed for reproducibility within this test run
	rand.Seed(42)
	opt, err := WithEvictionPolicy(PolicyRR)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	cache, err := NewCache(2, opt)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	cache.Set("a", []float32{1}, "A")
	cache.Set("b", []float32{2}, "B")
	// Insert "c", should evict either 'a' or 'b'
	cache.Set("c", []float32{3}, "C")
	// Exactly capacity keys should remain among {"a","b","c"}
	keys := []string{"a", "b", "c"}
	var count int
	for _, key := range keys {
		if _, ok := cache.Get(key); ok {
			count++
		}
	}
	if count != 2 {
		t.Errorf("RR: expected exactly 2 entries to remain, got %d", count)
	}
}
