// Package commands provides all CLI commands for the devops tool
package commands

import (
	"context"
	"fmt"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
	"github.com/raja-aiml/sematic-cache/devops/pkg/factory"
)

// RootCommand represents the base command
type RootCommand struct {
	factory    *factory.Factory
	rootCmd    *cobra.Command
	configFile string
}

// NewRootCommand creates a new root command
func NewRootCommand() (*RootCommand, error) {
	// Create default factory
	factoryConfig := factory.DefaultConfig()
	f, err := factory.NewFactory(factoryConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create factory: %w", err)
	}

	rc := &RootCommand{
		factory: f,
	}

	rc.rootCmd = &cobra.Command{
		Use:   "devops",
		Short: "DevOps automation toolkit",
		Long: `A comprehensive DevOps automation toolkit for Go projects.
		
This tool provides various utilities for development operations including:
- Development tool installation and management
- Docker and Kubernetes operations
- Task documentation generation
- System prerequisites validation`,
		PersistentPreRunE: rc.initializeConfig,
	}

	// Add persistent flags
	rc.rootCmd.PersistentFlags().StringVar(&rc.configFile, "config", "", "config file (default is $HOME/.devops.yaml)")
	rc.rootCmd.PersistentFlags().String("log-level", "info", "log level (debug, info, warn, error)")
	rc.rootCmd.PersistentFlags().Bool("no-color", false, "disable colored output")

	// Bind flags to viper
	viper.BindPFlag("log.level", rc.rootCmd.PersistentFlags().Lookup("log-level"))
	viper.BindPFlag("log.noColor", rc.rootCmd.PersistentFlags().Lookup("no-color"))

	// Add subcommands
	rc.addSubcommands()

	return rc, nil
}

// Execute runs the root command
func (rc *RootCommand) Execute() error {
	return rc.rootCmd.Execute()
}

// GetCommand returns the cobra command
func (rc *RootCommand) GetCommand() *cobra.Command {
	return rc.rootCmd
}

// GetFactory returns the factory instance
func (rc *RootCommand) GetFactory() *factory.Factory {
	return rc.factory
}

// initializeConfig initializes the configuration
func (rc *RootCommand) initializeConfig(cmd *cobra.Command, args []string) error {
	if rc.configFile != "" {
		viper.SetConfigFile(rc.configFile)
	} else {
		viper.SetConfigName(".devops")
		viper.SetConfigType("yaml")
		viper.AddConfigPath(".")
		viper.AddConfigPath("$HOME")
	}

	viper.SetEnvPrefix("DEVOPS")
	viper.AutomaticEnv()

	if err := viper.ReadInConfig(); err == nil {
		rc.factory.GetLogger().Info("Using config file: %s", viper.ConfigFileUsed())
	}

	// Update factory configuration based on viper settings
	if err := rc.updateFactoryConfig(); err != nil {
		return fmt.Errorf("failed to update factory config: %w", err)
	}

	return nil
}

// updateFactoryConfig updates factory configuration from viper
func (rc *RootCommand) updateFactoryConfig() error {
	// This would update the factory configuration based on viper settings
	// For now, this is a placeholder
	return nil
}

// addSubcommands adds all subcommands to the root command
func (rc *RootCommand) addSubcommands() {
	// Add version command
	rc.rootCmd.AddCommand(NewVersionCommand(rc.factory).GetCommand())

	// Add install command
	rc.rootCmd.AddCommand(NewInstallCommand(rc.factory).GetCommand())

	// Add precheck command
	rc.rootCmd.AddCommand(NewPrecheckCommand(rc.factory).GetCommand())

	// Add completion command
	rc.rootCmd.AddCommand(NewCompletionCommand(rc.factory).GetCommand())

	// Add docker command group
	rc.rootCmd.AddCommand(NewDockerCommand(rc.factory).GetCommand())

	// Add kubernetes command group
	rc.rootCmd.AddCommand(NewKubernetesCommand(rc.factory).GetCommand())

	// Add taskdoc command
	rc.rootCmd.AddCommand(NewTaskdocCommand(rc.factory).GetCommand())
}

// BaseCommand provides common functionality for all commands
type BaseCommand struct {
	factory *factory.Factory
	logger  interfaces.Logger
}

// NewBaseCommand creates a new base command
func NewBaseCommand(factory *factory.Factory) *BaseCommand {
	return &BaseCommand{
		factory: factory,
		logger:  factory.GetLogger(),
	}
}

// GetFactory returns the factory instance
func (bc *BaseCommand) GetFactory() *factory.Factory {
	return bc.factory
}

// GetLogger returns the logger instance
func (bc *BaseCommand) GetLogger() interfaces.Logger {
	return bc.logger
}

// GetHTTPClient returns the HTTP client instance
func (bc *BaseCommand) GetHTTPClient() interfaces.HTTPClient {
	return bc.factory.GetHTTPClient()
}

// GetOSUtil returns the OS utility instance
func (bc *BaseCommand) GetOSUtil() interfaces.OSUtil {
	return bc.factory.GetOSUtil()
}

// GetDockerClient returns the Docker client instance
func (bc *BaseCommand) GetDockerClient() (interfaces.DockerClient, error) {
	return bc.factory.GetDockerClient()
}

// GetKubernetesClient returns the Kubernetes client instance
func (bc *BaseCommand) GetKubernetesClient() (interfaces.KubernetesClient, error) {
	return bc.factory.GetKubernetesClient()
}

// RunWithContext runs a function with a context
func (bc *BaseCommand) RunWithContext(fn func(context.Context) error) error {
	ctx := context.Background()
	return fn(ctx)
}
