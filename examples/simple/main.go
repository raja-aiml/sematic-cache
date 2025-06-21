// Example demonstrating basic cache usage.
package main

import (
	"fmt"

	"github.com/raja-aiml/sematic-cache/core"
)

func main() {
	// Create an in-memory cache with capacity of 100
	cache, err := core.NewCache(100)
	if err != nil {
		panic(err)
	}

	// Store a prompt-response pair
	cache.Set("What is AI?", nil, "Artificial Intelligence is...")

	// Retrieve the cached response
	if answer, found := cache.Get("What is AI?"); found {
		fmt.Println("Cached answer:", answer)
	}

	// Attempt to retrieve a non-existent prompt
	if answer, found := cache.Get("What is ML?"); !found {
		fmt.Println("No cached answer for 'What is ML?'")
	} else {
		fmt.Println("Cached answer:", answer)
	}

}
