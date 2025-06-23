package cmd

import (
	"context"

	"github.com/spf13/cobra"

	"github.com/raja-aiml/sematic-cache/devops/internal/logger"
	"github.com/raja-aiml/sematic-cache/devops/pkg/factory"
)

var (
	toolsInstallerSkipConfirm bool
	toolsInstallerDebug       bool
)

// toolsInstallerCmd represents the tools-installer command
var toolsInstallerCmd = &cobra.Command{
	Use:   "tools-installer",
	Short: "Install development tools using the comprehensive tool installer",
	Long: `Install development tools using the comprehensive tool installer framework.

This command provides a complete tool installation system with:
- Automatic platform detection
- Version management
- Installation verification
- Comprehensive tool catalog

Supported tools:
- task (build automation)
- golangci-lint (Go linter)
- gofumpt (stricter gofmt)
- mockgen (Go mock generator)
- k3d (local Kubernetes)
- helm (package manager)
- kustomize (config management)`,
	RunE: runToolsInstaller,
}

func runToolsInstaller(cmd *cobra.Command, args []string) error {
	log := logger.New()

	if toolsInstallerDebug {
		log.SetLevel(logger.DebugLevel)
	}

	log.Info("Starting comprehensive tool installation...")

	// Create factory with default config
	factoryConfig := factory.DefaultConfig()
	f, err := factory.NewFactory(factoryConfig)
	if err != nil {
		return err
	}

	// Create registry and register tools
	registry, err := f.CreateToolRegistry()
	if err != nil {
		return err
	}

	// Register standard tools
	if err := registerStandardTools(registry, f); err != nil {
		return err
	}

	ctx := context.Background()
	if err := registry.InstallAll(ctx); err != nil {
		log.Error("Installation failed: %v", err)
		return err
	}

	log.Success("All tools installed successfully!")
	return nil
}

func registerStandardTools(registry interface{}, f *factory.Factory) error {
	// The factory already creates and registers standard tools
	// This function is not needed anymore since CreateToolRegistry() handles it
	log := logger.New()
	log.Info("Standard development tools already registered via factory")
	return nil
}

func init() {
	toolsInstallerCmd.Flags().BoolVar(&toolsInstallerSkipConfirm, "skip-confirmation", false, "Skip installation confirmation")
	toolsInstallerCmd.Flags().BoolVar(&toolsInstallerDebug, "debug", false, "Enable debug logging")
}
