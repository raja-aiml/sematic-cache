package core

import "testing"

func TestCacheSetGet(t *testing.T) {
	c := NewCache()
	c.Set("hello", "world")
	if val, ok := c.Get("hello"); !ok || val != "world" {
		t.Fatalf("expected world, got %v, found %v", val, ok)
	}
}
