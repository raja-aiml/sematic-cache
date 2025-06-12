package core

import "testing"

func TestCacheSetGet(t *testing.T) {
	c := NewCache()
	c.Set("hello", "world")
	if val, ok := c.Get("hello"); !ok || val != "world" {
		t.Fatalf("expected world, got %v, found %v", val, ok)
	}
}

func TestCacheConcurrent(t *testing.T) {
	c := NewCache()
	done := make(chan struct{})
	go func() {
		for i := 0; i < 1000; i++ {
			c.Set("k", "v")
		}
		close(done)
	}()
	for i := 0; i < 1000; i++ {
		c.Get("k")
	}
	<-done
}
