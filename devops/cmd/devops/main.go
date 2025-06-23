// Package main provides the entry point for the devops CLI tool
package main

import (
	"fmt"
	"os"

	"github.com/raja-aiml/sematic-cache/devops/cmd/devops/commands"
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
	// Create root command
	rootCmd, err := commands.NewRootCommand()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing command: %v\n", err)
		os.Exit(1)
	}

	// Set version information
	commands.SetVersionInfo(version, commit, date, builtBy, goVersion)

	// Execute command
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
