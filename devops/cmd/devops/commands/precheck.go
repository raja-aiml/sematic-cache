// Package commands provides the precheck command
package commands

import (
	"context"
	"fmt"
	"runtime"
	"strings"

	"github.com/spf13/cobra"

	"github.com/raja-aiml/sematic-cache/devops/pkg/factory"
)

// PrecheckCommand handles system prerequisites checking
type PrecheckCommand struct {
	*BaseCommand
	cmd *cobra.Command

	// Command options
	verbose bool
}

// NewPrecheckCommand creates a new precheck command
func NewPrecheckCommand(factory *factory.Factory) *PrecheckCommand {
	pc := &PrecheckCommand{
		BaseCommand: NewBaseCommand(factory),
	}

	pc.cmd = &cobra.Command{
		Use:   "precheck",
		Short: "Check system prerequisites",
		Long: `Check if all required system prerequisites are met.

This command verifies:
- Required commands are available (go, git, docker, etc.)
- Go version meets minimum requirements
- Docker daemon is running (if applicable)
- Kubernetes context is configured (if applicable)`,
		Example: `  # Run basic prerequisite check
  devops precheck

  # Run verbose check with detailed output
  devops precheck --verbose`,
		RunE: pc.run,
	}

	// Add flags
	pc.cmd.Flags().BoolVarP(&pc.verbose, "verbose", "v", false, "Enable verbose output")

	return pc
}

// GetCommand returns the cobra command
func (pc *PrecheckCommand) GetCommand() *cobra.Command {
	return pc.cmd
}

// run executes the precheck command
func (pc *PrecheckCommand) run(cmd *cobra.Command, args []string) error {
	return pc.RunWithContext(func(ctx context.Context) error {
		pc.logger.Info("Running system prerequisite checks...")

		// Check required commands
		if err := pc.checkRequiredCommands(); err != nil {
			return err
		}

		// Check Go version
		if err := pc.checkGoVersion(); err != nil {
			return err
		}

		// Check Docker
		if err := pc.checkDocker(ctx); err != nil {
			// Docker is optional, just warn
			pc.logger.Warning("Docker check failed: %v", err)
		}

		// Check Kubernetes
		if err := pc.checkKubernetes(ctx); err != nil {
			// Kubernetes is optional, just warn
			pc.logger.Warning("Kubernetes check failed: %v", err)
		}

		pc.logger.Success("All prerequisite checks passed!")
		return nil
	})
}

// checkRequiredCommands checks if required commands are available
func (pc *PrecheckCommand) checkRequiredCommands() error {
	pc.logger.Info("Checking required commands...")

	required := []string{"go", "git"}
	osUtil := pc.GetOSUtil()

	missing, err := osUtil.VerifyCommands(required)
	if err != nil {
		return fmt.Errorf("command check failed: %w", err)
	}

	if len(missing) > 0 {
		return fmt.Errorf("missing required commands: %v", missing)
	}

	// Check optional commands
	optional := []string{"docker", "kubectl", "helm", "kustomize"}
	for _, cmd := range optional {
		if osUtil.IsCommandAvailable(cmd) {
			if pc.verbose {
				pc.logger.Success("✓ %s is available", cmd)
			}
		} else {
			if pc.verbose {
				pc.logger.Warning("✗ %s is not available (optional)", cmd)
			}
		}
	}

	pc.logger.Success("All required commands are available")
	return nil
}

// checkGoVersion checks if Go version meets requirements
func (pc *PrecheckCommand) checkGoVersion() error {
	pc.logger.Info("Checking Go version...")

	// Get Go version
	goVersion := runtime.Version()
	if pc.verbose {
		pc.logger.Info("Go version: %s", goVersion)
	}

	// Extract version number
	version := strings.TrimPrefix(goVersion, "go")
	parts := strings.Split(version, ".")

	if len(parts) < 2 {
		return fmt.Errorf("unable to parse Go version: %s", goVersion)
	}

	// Check minimum version (Go 1.19+)
	major := parts[0]
	minor := parts[1]

	if major != "1" {
		return fmt.Errorf("unsupported Go major version: %s", major)
	}

	minorInt := 0
	fmt.Sscanf(minor, "%d", &minorInt)

	if minorInt < 19 {
		return fmt.Errorf("Go version 1.19 or higher is required, found: %s", goVersion)
	}

	pc.logger.Success("Go version %s meets requirements", goVersion)
	return nil
}

// checkDocker checks if Docker is available and running
func (pc *PrecheckCommand) checkDocker(ctx context.Context) error {
	pc.logger.Info("Checking Docker...")

	osUtil := pc.GetOSUtil()
	if !osUtil.IsCommandAvailable("docker") {
		return fmt.Errorf("docker command not found")
	}

	// Check if Docker daemon is running
	dockerClient, err := pc.GetDockerClient()
	if err != nil {
		return fmt.Errorf("failed to create Docker client: %w", err)
	}

	if !dockerClient.IsRunning(ctx) {
		return fmt.Errorf("Docker daemon is not running")
	}

	pc.logger.Success("Docker is available and running")
	return nil
}

// checkKubernetes checks if Kubernetes is configured
func (pc *PrecheckCommand) checkKubernetes(ctx context.Context) error {
	pc.logger.Info("Checking Kubernetes...")

	osUtil := pc.GetOSUtil()
	if !osUtil.IsCommandAvailable("kubectl") {
		return fmt.Errorf("kubectl command not found")
	}

	// Check if kubeconfig is available
	k8sClient, err := pc.GetKubernetesClient()
	if err != nil {
		return fmt.Errorf("failed to create Kubernetes client: %w", err)
	}

	// Get current context
	currentContext, err := k8sClient.GetCurrentContext()
	if err != nil {
		return fmt.Errorf("failed to get current context: %w", err)
	}

	if currentContext == "" {
		return fmt.Errorf("no Kubernetes context configured")
	}

	if pc.verbose {
		pc.logger.Info("Current Kubernetes context: %s", currentContext)
	}

	pc.logger.Success("Kubernetes is configured with context: %s", currentContext)
	return nil
}
