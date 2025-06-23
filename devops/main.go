// Package main provides the entry point for the devops CLI tool
package main

import (
	"fmt"
	"os"

	"github.com/raja-aiml/sematic-cache/devops/internal/devops/cmd"
)

// version information (set at build time)
var (
	version   = "dev"
	commit    = "none"
	date      = "unknown"
	builtBy   = "unknown"
	goVersion = "unknown"
)

func main() {
	// Set version information
	cmd.SetVersionInfo(version, commit, date, builtBy, goVersion)

	// Execute command
	if err := cmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
