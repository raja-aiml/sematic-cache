package main

import (
	"fmt"
	"os"
	"runtime"

	"github.com/raja-aiml/sematic-cache/deploy/local/cmd"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/config"
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

	// Add global flags
	rootCmd.PersistentFlags().String("config-dir", "", "Config directory path (default: ./config or ../config)")
	rootCmd.PersistentFlags().String("env-file", "", "Path to environment file (default: <config-dir>/blueprint.env)")

	// Add commands
	rootCmd.AddCommand(cmd.ClusterCmd())
	rootCmd.AddCommand(cmd.DevCmd())
	rootCmd.AddCommand(cmd.WorkflowCmd())
	rootCmd.AddCommand(cmd.CompositeTestCmd())
	rootCmd.AddCommand(cmd.DebugCmd())
	rootCmd.AddCommand(cmd.TestCmd())
	rootCmd.AddCommand(cmd.ValidateCmd())
	rootCmd.AddCommand(cmd.ManifestCmd())
	rootCmd.AddCommand(cmd.AgentCmd())
	rootCmd.AddCommand(cmd.DocsCmd(rootCmd, version))
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
	cmd := &cobra.Command{
		Use:   "config",
		Short: "Configuration management commands",
		Long:  `Manage configuration files, validate settings, and export configurations.`,
	}

	// Add subcommands
	cmd.AddCommand(configShowCmd())
	cmd.AddCommand(configPathCmd())

	return cmd
}

func configShowCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "show",
		Short: "Show current configuration",
		RunE: func(cmd *cobra.Command, args []string) error {
			paths, err := config.ResolveConfigPaths(cmd)
			if err != nil {
				return fmt.Errorf("failed to resolve config paths: %w", err)
			}

			fmt.Printf("Configuration:\n")
			fmt.Printf("  Config Directory: %s\n", paths.ConfigDir)
			fmt.Printf("  Environment File: %s\n", paths.EnvFile)

			// Check if files exist
			if _, err := os.Stat(paths.ConfigDir); err == nil {
				fmt.Printf("  Config Dir Status: ✓ Exists\n")
			} else {
				fmt.Printf("  Config Dir Status: ✗ Not found\n")
			}

			if _, err := os.Stat(paths.EnvFile); err == nil {
				fmt.Printf("  Env File Status: ✓ Exists\n")
			} else if _, err := os.Stat(paths.EnvFile + ".example"); err == nil {
				fmt.Printf("  Env File Status: ⚠ Using .example file\n")
			} else {
				fmt.Printf("  Env File Status: ✗ Not found\n")
			}

			return nil
		},
	}
}

func configPathCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "path",
		Short: "Show resolved configuration path",
		RunE: func(cmd *cobra.Command, args []string) error {
			paths, err := config.ResolveConfigPaths(cmd)
			if err != nil {
				return fmt.Errorf("failed to resolve config paths: %w", err)
			}

			// Output just the path for scripting
			fmt.Println(paths.ConfigDir)
			return nil
		},
	}
}
