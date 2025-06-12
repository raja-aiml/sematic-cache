package main

import (
	"fmt"

	"github.com/raja-aiml/sematic-cache/go/core"
)

func main() {
	cache := core.NewCache(10)
	cache.Set("hello", nil, "world")
	if val, ok := cache.Get("hello"); ok {
		fmt.Println("cached:", val)
	} else {
		fmt.Println("not found")
	}
}
