package main

import (
	"log"
	"os"

	"github.com/raja-aiml/sematic-cache/cmd"
)

func main() {
	if err := cmd.Run(); err != nil {
		log.Printf("Server error: %v\n", err)
		os.Exit(1)
	}
}
