package cmd

import (
	"context"

	"github.com/spf13/cobra"

	"github.com/raja-aiml/sematic-cache/devops/internal/logger"
	"github.com/raja-aiml/sematic-cache/devops/pkg/factory"
)

var (
	installToolsSkipConfirm bool
)

// installToolsCmd represents the install-tools command
var installToolsCmd = &cobra.Command{
	Use:   "install-tools",
	Short: "Install development tools using the interface-based installer",
	Long: `Install development tools using the new interface-based installer framework.

This command uses the modern interface-based architecture for tool installation
with proper dependency injection and comprehensive testing.`,
	RunE: runInstallTools,
}

func runInstallTools(cmd *cobra.Command, args []string) error {
	log := logger.New()

	log.Info("Starting tool installation using interface-based installer...")

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

	ctx := context.Background()
	if err := registry.InstallAll(ctx); err != nil {
		log.Error("Installation failed: %v", err)
		return err
	}

	log.Success("All tools installed successfully!")
	return nil
}

func init() {
	installToolsCmd.Flags().BoolVar(&installToolsSkipConfirm, "skip-confirmation", false, "Skip installation confirmation")
}