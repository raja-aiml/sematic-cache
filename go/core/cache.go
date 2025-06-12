package core

import "sync"

// Cache provides a concurrent in-memory cache for prompts and answers.
type Cache struct {
	mu   sync.RWMutex
	data map[string]string
}

// NewCache creates a new Cache instance.
func NewCache() *Cache {
	return &Cache{data: make(map[string]string)}
}

// Set stores an answer for the given prompt.
func (c *Cache) Set(prompt, answer string) {
	c.mu.Lock()
	c.data[prompt] = answer
	c.mu.Unlock()
}

// Get returns the cached answer for a prompt and whether it was found.
func (c *Cache) Get(prompt string) (string, bool) {
	c.mu.RLock()
	val, ok := c.data[prompt]
	c.mu.RUnlock()
	return val, ok
}
