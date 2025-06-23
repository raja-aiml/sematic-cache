// Package commands provides the install command implementation
package commands

import (
	"context"
	"fmt"

	"github.com/spf13/cobra"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
	"github.com/raja-aiml/sematic-cache/devops/pkg/factory"
	"github.com/raja-aiml/sematic-cache/devops/pkg/tools"
)

// InstallCommand handles tool installation
type InstallCommand struct {
	*BaseCommand
	cmd *cobra.Command

	// Command options
	skipConfirm    bool
	parallel       bool
	maxConcurrency int
	force          bool
	tools          []string
}

// NewInstallCommand creates a new install command
func NewInstallCommand(factory *factory.Factory) *InstallCommand {
	ic := &InstallCommand{
		BaseCommand: NewBaseCommand(factory),
	}

	ic.cmd = &cobra.Command{
		Use:   "install",
		Short: "Install development tools",
		Long: `Install required development tools for Go projects.

This command installs various development tools commonly used in Go projects
including linters, formatters, build tools, and container orchestration tools.

The tools are installed using their official installation methods:
- Binary downloads for pre-built tools
- 'go install' for Go-based tools
- SDK-based installation where available`,
		Example: `  # Install all tools
  devops install

  # Install specific tools
  devops install --tools task,golangci-lint

  # Install without confirmation
  devops install --skip-confirm

  # Force reinstall all tools
  devops install --force

  # Install tools in parallel
  devops install --parallel --max-concurrency 4`,
		RunE: ic.run,
	}

	// Add flags
	ic.cmd.Flags().BoolVarP(&ic.skipConfirm, "skip-confirm", "y", false, "Skip confirmation prompt")
	ic.cmd.Flags().BoolVar(&ic.parallel, "parallel", false, "Install tools in parallel")
	ic.cmd.Flags().IntVar(&ic.maxConcurrency, "max-concurrency", 4, "Maximum number of parallel installations")
	ic.cmd.Flags().BoolVar(&ic.force, "force", false, "Force reinstall even if already installed")
	ic.cmd.Flags().StringSliceVar(&ic.tools, "tools", []string{}, "Specific tools to install (comma-separated)")

	return ic
}

// GetCommand returns the cobra command
func (ic *InstallCommand) GetCommand() *cobra.Command {
	return ic.cmd
}

// run executes the install command
func (ic *InstallCommand) run(cmd *cobra.Command, args []string) error {
	return ic.RunWithContext(func(ctx context.Context) error {
		// Create tool registry
		registry, err := ic.createRegistry()
		if err != nil {
			return fmt.Errorf("failed to create tool registry: %w", err)
		}

		// Check prerequisites
		if err := ic.checkPrerequisites(); err != nil {
			return fmt.Errorf("prerequisite check failed: %w", err)
		}

		// Install tools
		if ic.parallel {
			return ic.installParallel(ctx, registry)
		}

		return ic.installSequential(ctx, registry)
	})
}

// createRegistry creates a tool registry with selected tools
func (ic *InstallCommand) createRegistry() (interfaces.ToolRegistry, error) {
	if len(ic.tools) > 0 {
		// Create registry with specific tools
		return ic.createCustomRegistry()
	}

	// Create registry with all default tools
	return ic.factory.CreateToolRegistry()
}

// createCustomRegistry creates a registry with specific tools
func (ic *InstallCommand) createCustomRegistry() (interfaces.ToolRegistry, error) {
	registry := tools.NewRegistry(ic.logger)

	// Get all available tools
	allTools, err := ic.factory.CreateToolRegistry()
	if err != nil {
		return nil, err
	}

	// Register only selected tools
	for _, toolName := range ic.tools {
		tool, err := allTools.Get(toolName)
		if err != nil {
			ic.logger.Warning("Unknown tool: %s", toolName)
			continue
		}

		if err := registry.Register(tool); err != nil {
			return nil, fmt.Errorf("failed to register %s: %w", toolName, err)
		}
	}

	if len(registry.List()) == 0 {
		return nil, fmt.Errorf("no valid tools selected")
	}

	return registry, nil
}

// checkPrerequisites checks system prerequisites
func (ic *InstallCommand) checkPrerequisites() error {
	ic.logger.Info("Checking prerequisites...")

	// Check for required commands
	required := []string{"go", "git"}

	missing, err := ic.GetOSUtil().VerifyCommands(required)
	if err != nil {
		return err
	}

	if len(missing) > 0 {
		return fmt.Errorf("missing required commands: %v", missing)
	}

	// Check Go version
	output, err := ic.runCommand("go", "version")
	if err != nil {
		return fmt.Errorf("failed to check Go version: %w", err)
	}

	ic.logger.Success("Prerequisites satisfied: %s", output)
	return nil
}

// installSequential installs tools one by one
func (ic *InstallCommand) installSequential(ctx context.Context, registry interfaces.ToolRegistry) error {
	tools := registry.List()

	ic.logger.Info("Installing %d tools...", len(tools))

	// Show tools to install
	if !ic.skipConfirm {
		if err := ic.confirmInstallation(tools); err != nil {
			return err
		}
	}

	// Install each tool
	for _, tool := range tools {
		if err := ic.installTool(ctx, tool); err != nil {
			return err
		}
	}

	ic.logger.Success("All tools installed successfully!")

	// Validate installations
	return registry.ValidateAll()
}

// installParallel installs tools in parallel
func (ic *InstallCommand) installParallel(ctx context.Context, registry interfaces.ToolRegistry) error {
	opts := interfaces.InstallOptions{
		Parallel:       true,
		MaxConcurrency: ic.maxConcurrency,
		Force:          ic.force,
		SkipValidation: false,
	}

	return registry.InstallAllWithOptions(ctx, opts)
}

// installTool installs a single tool
func (ic *InstallCommand) installTool(ctx context.Context, tool interfaces.ToolInstaller) error {
	if !ic.force && tool.IsInstalled() {
		version, _ := tool.GetInstalledVersion()
		ic.logger.Success("%s is already installed: %s", tool.Name(), version)
		return nil
	}

	ic.logger.Info("Installing %s (%s)...", tool.Name(), tool.Description())

	if err := tool.Install(ctx); err != nil {
		return fmt.Errorf("failed to install %s: %w", tool.Name(), err)
	}

	// Validate installation
	if err := tool.Validate(); err != nil {
		return fmt.Errorf("validation failed for %s: %w", tool.Name(), err)
	}

	return nil
}

// confirmInstallation shows tools to install and asks for confirmation
func (ic *InstallCommand) confirmInstallation(tools []interfaces.ToolInstaller) error {
	ic.logger.Info("The following tools will be installed:")

	for _, tool := range tools {
		if !ic.force && tool.IsInstalled() {
			continue
		}
		fmt.Printf("  - %s (%s) version %s\n", tool.Name(), tool.Description(), tool.Version())
	}

	fmt.Print("\nProceed with installation? [y/N] ")

	var response string
	fmt.Scanln(&response)

	if response != "y" && response != "Y" {
		return fmt.Errorf("installation cancelled")
	}

	return nil
}

// runCommand runs a command and returns its output
func (ic *InstallCommand) runCommand(name string, args ...string) (string, error) {
	// This would use the command runner interface
	// For now, this is a placeholder
	return "", nil
}
