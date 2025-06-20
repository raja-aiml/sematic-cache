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
	Use:   "iaac",
	Short: "Infrastructure as Code tool for local Kubernetes deployments",
	Long: `A comprehensive CLI tool for managing application deployments in local Kubernetes environments.
This tool provides infrastructure-as-code capabilities using k3d, Docker SDK, and Kubernetes client-go.`,
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
			if _, err := fmt.Fprintf(cmd.OutOrStdout(), "IaaC - Infrastructure as Code Tool\n"); err != nil {
				cmd.PrintErrf("Failed to write output: %v\n", err)
			}
			if _, err := fmt.Fprintf(cmd.OutOrStdout(), "  Version:    %s\n", version); err != nil {
				cmd.PrintErrf("Failed to write output: %v\n", err)
			}
			if _, err := fmt.Fprintf(cmd.OutOrStdout(), "  Built:      %s\n", buildTime); err != nil {
				cmd.PrintErrf("Failed to write output: %v\n", err)
			}
			if _, err := fmt.Fprintf(cmd.OutOrStdout(), "  Git Commit: %s\n", gitCommit); err != nil {
				cmd.PrintErrf("Failed to write output: %v\n", err)
			}
			if _, err := fmt.Fprintf(cmd.OutOrStdout(), "  Go Version: %s\n", runtime.Version()); err != nil {
				cmd.PrintErrf("Failed to write output: %v\n", err)
			}
			if _, err := fmt.Fprintf(cmd.OutOrStdout(), "  OS/Arch:    %s/%s\n", runtime.GOOS, runtime.GOARCH); err != nil {
				cmd.PrintErrf("Failed to write output: %v\n", err)
			}
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
