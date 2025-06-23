package cmd

import (
	"context"
	"fmt"
	"strings"

	"github.com/raja-aiml/sematic-cache/devops/internal/logger"
	"github.com/raja-aiml/sematic-cache/devops/pkg/factory"
	"github.com/spf13/cobra"
)

// installCmd represents the install command
var installCmd = &cobra.Command{
	Use:   "install [tool...]",
	Short: "Install development tools",
	Long: `Install one or more development tools required for the project.

Available tools:
- task: Task runner for build automation
- golangci-lint: Go linter
- k3d: Local Kubernetes clusters
- helm: Kubernetes package manager
- kustomize: Kubernetes configuration management
- kubectl: Kubernetes CLI (if not included with Docker Desktop)
- all: Install all tools

The installer will automatically detect your OS and architecture.`,
	Example: `  # Install all tools
  devops install

  # Install specific tools
  devops install task k3d

  # Install without confirmation prompt
  devops install --yes

  # List available tools
  devops install --list`,
	RunE: runInstall,
}

var (
	installSkipConfirm bool
	installList        bool
)

func init() {
	installCmd.Flags().BoolVarP(&installSkipConfirm, "yes", "y", false, "Skip confirmation prompt")
	installCmd.Flags().BoolVarP(&installList, "list", "l", false, "List available tools")
}

func runInstall(cmd *cobra.Command, args []string) error {
	log := logger.New()

	// Define available tools
	availableTools := map[string]string{
		"task":          "Task (build automation)",
		"golangci-lint": "golangci-lint (Go linter)",
		"k3d":           "k3d (local Kubernetes)",
		"helm":          "Helm (package manager)",
		"kustomize":     "Kustomize (config management)",
		"kubectl":       "kubectl (Kubernetes CLI)",
	}

	// List mode
	if installList {
		log.Info("Available tools:")
		for name, desc := range availableTools {
			fmt.Printf("  %s - %s\n", name, desc)
		}
		return nil
	}

	// Determine which tools to install
	var toolsToInstall []string

	if len(args) == 0 || (len(args) == 1 && args[0] == "all") {
		// Install all tools using factory approach
		log.Info("Installing all development tools...")

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
		return registry.InstallAll(ctx)
	}

	// Install specific tools
	for _, arg := range args {
		toolName := strings.ToLower(arg)
		if _, ok := availableTools[toolName]; !ok {
			log.Error("Unknown tool: %s", toolName)
			log.Info("Run 'devops install --list' to see available tools")
			return fmt.Errorf("unknown tool: %s", toolName)
		}
		toolsToInstall = append(toolsToInstall, toolName)
	}

	// Install specific tools using factory approach
	log.Info("Installing selected tools: %v", toolsToInstall)

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

	// Install specific tools
	for _, toolName := range toolsToInstall {
		if err := registry.Install(ctx, toolName); err != nil {
			log.Error("Failed to install %s: %v", toolName, err)
			return err
		}
	}

	log.Success("Selected tools installed successfully!")
	return nil
}
