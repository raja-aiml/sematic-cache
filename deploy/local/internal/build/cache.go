package build

import (
	"context"
	"sync"
	"time"
)

// MemoryCache implements BuildCache using in-memory storage
type MemoryCache struct {
	mu      sync.RWMutex
	entries map[string]*CacheEntry
}

// NewMemoryCache creates a new in-memory cache
func NewMemoryCache() *MemoryCache {
	return &MemoryCache{
		entries: make(map[string]*CacheEntry),
	}
}

// Get retrieves a cache entry
func (c *MemoryCache) Get(ctx context.Context, key string) (*CacheEntry, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, exists := c.entries[key]
	if !exists {
		return nil, nil
	}

	// Check if expired
	if time.Now().After(entry.ExpiresAt) {
		return nil, nil
	}

	return entry, nil
}

// Set stores a cache entry
func (c *MemoryCache) Set(ctx context.Context, key string, entry *CacheEntry) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.entries[key] = entry
	return nil
}

// Delete removes a cache entry
func (c *MemoryCache) Delete(ctx context.Context, key string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.entries, key)
	return nil
}

// Clear removes all cache entries
func (c *MemoryCache) Clear(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.entries = make(map[string]*CacheEntry)
	return nil
}

// Cleanup removes expired entries
func (c *MemoryCache) Cleanup() {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	for key, entry := range c.entries {
		if now.After(entry.ExpiresAt) {
			delete(c.entries, key)
		}
	}
}
