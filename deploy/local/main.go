package main

import (
	"fmt"
	"os"
	"runtime"

	"github.com/raja-aiml/sematic-cache/deploy/local/cmd"
	"github.com/spf13/cobra"
)

// Build information - set via ldflags
var (
	version   = "dev"
	buildTime = "unknown"
	gitCommit = "unknown"
)

var rootCmd = &cobra.Command{
	Use:   "semantic-cache-deploy",
	Short: "Semantic Cache deployment tools for local Kubernetes development",
	Long: `A comprehensive CLI tool for managing semantic cache deployments in local Kubernetes environments.
This tool replaces shell scripts with a robust Go implementation using k3d SDK.`,
	Version: fmt.Sprintf("%s (built %s, commit %s)", version, buildTime, gitCommit),
}

func init() {
	// Disable completion command
	rootCmd.CompletionOptions.DisableDefaultCmd = true
	
	// Add commands
	rootCmd.AddCommand(cmd.ClusterCmd())
	rootCmd.AddCommand(cmd.DevCmd())
	rootCmd.AddCommand(cmd.WorkflowCmd())
	rootCmd.AddCommand(cmd.CompositeTestCmd())
	rootCmd.AddCommand(cmd.DebugCmd())
	rootCmd.AddCommand(versionCmd())
	rootCmd.AddCommand(configCmd())
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func versionCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print version information",
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Printf("Semantic Cache Deploy\n")
			fmt.Printf("  Version:    %s\n", version)
			fmt.Printf("  Built:      %s\n", buildTime)
			fmt.Printf("  Git Commit: %s\n", gitCommit)
			fmt.Printf("  Go Version: %s\n", runtime.Version())
			fmt.Printf("  OS/Arch:    %s/%s\n", runtime.GOOS, runtime.GOARCH)
		},
	}
}

func configCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "config",
		Short: "Configuration management commands",
		Long:  `Manage configuration files, validate settings, and export configurations.`,
	}
}