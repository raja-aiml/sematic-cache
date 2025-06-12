package core

// Cache provides a simple in-memory cache for prompts and answers.
type Cache struct {
	data map[string]string
}

// NewCache creates a new Cache instance.
func NewCache() *Cache {
	return &Cache{data: make(map[string]string)}
}

// Set stores an answer for the given prompt.
func (c *Cache) Set(prompt, answer string) {
	c.data[prompt] = answer
}

// Get returns the cached answer for a prompt and whether it was found.
func (c *Cache) Get(prompt string) (string, bool) {
	val, ok := c.data[prompt]
	return val, ok
}
