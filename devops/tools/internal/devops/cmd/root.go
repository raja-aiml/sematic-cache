package cmd

import (
	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "devops",
	Short: "DevOps tooling for semantic-cache project",
	Long: `DevOps provides various development and operations tools for the semantic-cache project.

This includes:
- taskdoc: Generate documentation for Taskfiles
- Future tools for build, deploy, and infrastructure management`,
	SilenceUsage: true,
}

// Execute runs the root command
func Execute() error {
	return rootCmd.Execute()
}

func init() {
	// Add subcommands
	rootCmd.AddCommand(taskdocCmd)
	rootCmd.AddCommand(versionCmd)
	rootCmd.AddCommand(validateCmd)
}
