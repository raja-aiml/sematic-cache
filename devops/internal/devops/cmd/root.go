package cmd

import (
	"github.com/spf13/cobra"
)

// version information (set by main)
var (
	Version   = "dev"
	Commit    = "none"
	Date      = "unknown"
	BuiltBy   = "unknown"
	GoVersion = "unknown"
)

// SetVersionInfo sets the version information
func SetVersionInfo(version, commit, date, builtBy, goVersion string) {
	Version = version
	Commit = commit
	Date = date
	BuiltBy = builtBy
	GoVersion = goVersion
}

var rootCmd = &cobra.Command{
	Use:   "devops",
	Short: "DevOps tooling for semantic-cache project",
	Long: `DevOps provides various development and operations tools for the semantic-cache project.

This includes:
- precheck: Check system prerequisites and dependencies
- install: Install development tools (legacy CLI support)
- install-tools: Install tools using modern interface-based architecture
- tools-installer: Comprehensive tool installer with advanced features
- taskdoc: Generate documentation for Taskfiles
- validate: Validate Taskfile syntax
- version: Show version information

All tool installation commands now use SDK-first approach instead of CLI commands.`,
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
	rootCmd.AddCommand(installToolsCmd)
	rootCmd.AddCommand(toolsInstallerCmd)
	rootCmd.AddCommand(taskdocCmd)
	rootCmd.AddCommand(validateCmd)
	rootCmd.AddCommand(versionCmd)
	rootCmd.AddCommand(completionCmd)
}
