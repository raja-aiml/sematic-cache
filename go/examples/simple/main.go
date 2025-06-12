package main

import (
	"fmt"

	"github.com/raja-aiml/sematic-cache/go/core"
)

func main() {
	cache := core.NewCache()
	cache.Set("hello", "world")
	if val, ok := cache.Get("hello"); ok {
		fmt.Println("cached:", val)
	} else {
		fmt.Println("not found")
	}
}
