package main

import (
	"github.com/raja-aiml/sematic-cache/cmd"
	"github.com/raja-aiml/sematic-cache/internal/logger"
)

func main() {
	// Ensure logs are flushed on exit
	defer func() {
		if err := logger.Sync(); err != nil {
			// Best effort - ignore sync errors
			_ = err
		}
	}()

	if err := cmd.Run(); err != nil {
		logger.Fatal("Server error", logger.Fields{"error": err.Error()})
	}
}
