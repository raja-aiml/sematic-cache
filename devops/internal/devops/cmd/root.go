package cmd

import (
	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "devops",
	Short: "DevOps tooling for semantic-cache project",
	Long: `DevOps provides various development and operations tools for the semantic-cache project.

This includes:
- precheck: Check system prerequisites and dependencies
- install: Install development tools
- taskdoc: Generate documentation for Taskfiles
- validate: Validate Taskfile syntax
- version: Show version information`,
	SilenceUsage: true,
}

// Execute runs the root command
func Execute() error {
	return rootCmd.Execute()
}

func init() {
	// Add subcommands
	rootCmd.AddCommand(precheckCmd)
	rootCmd.AddCommand(installCmd)
	rootCmd.AddCommand(taskdocCmd)
	rootCmd.AddCommand(validateCmd)
	rootCmd.AddCommand(versionCmd)
	rootCmd.AddCommand(completionCmd)
}
