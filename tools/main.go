package main

import (
	"os"

	"github.com/raja-aiml/sematic-cache/tools/cmd"
)

func main() {
	// Execute the CLI directly - it will handle everything including config loading
	if err := cmd.ExecuteWithArgs(); err != nil {
		os.Exit(1)
	}
}
