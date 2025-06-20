package cmd

import (
	"context"
	"fmt"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/kubernetes"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
	"github.com/spf13/cobra"
)

// BaseCommand provides common functionality for all commands
type BaseCommand struct {
	Logger    *utils.Logger
	K8sClient *kubernetes.Client

	// Configuration
	ClusterName string
	Namespace   string
}

// NewBaseCommand creates a new base command
func NewBaseCommand(loggerPrefix string) *BaseCommand {
	return &BaseCommand{
		Logger: utils.NewLogger(loggerPrefix),
	}
}

// Initialize sets up the base command with Kubernetes client
func (b *BaseCommand) Initialize(ctx context.Context) error {
	if b.K8sClient == nil {
		client, err := kubernetes.GetDefaultClient()
		if err != nil {
			return fmt.Errorf("failed to initialize k8s client: %w", err)
		}
		b.K8sClient = client
	}
	return nil
}

// AddCommonFlags adds common flags to a cobra command
func (b *BaseCommand) AddCommonFlags(cmd *cobra.Command) {
	// These flags can be overridden by specific commands
	if b.ClusterName != "" {
		cmd.PersistentFlags().StringVarP(&b.ClusterName, "cluster", "c", b.ClusterName, "Cluster name")
	}
	if b.Namespace != "" {
		cmd.PersistentFlags().StringVarP(&b.Namespace, "namespace", "n", b.Namespace, "Namespace")
	}
}

// Execute wraps command execution with common error handling
func (b *BaseCommand) Execute(fn func() error) error {
	ctx := context.Background()

	// Initialize if needed
	if err := b.Initialize(ctx); err != nil {
		b.Logger.Error("Initialization failed: %v", err)
		return err
	}

	// Execute the actual command
	if err := fn(); err != nil {
		b.Logger.Error("Command failed: %v", err)
		return err
	}

	return nil
}

// CommandRunner provides a standard interface for command execution
type CommandRunner interface {
	Run(ctx context.Context) error
}

// CommandBuilder helps build commands with common patterns
type CommandBuilder struct {
	Use   string
	Short string
	Long  string
	RunE  func(cmd *cobra.Command, args []string) error
}

// Build creates a cobra command from the builder
func (cb *CommandBuilder) Build() *cobra.Command {
	return &cobra.Command{
		Use:   cb.Use,
		Short: cb.Short,
		Long:  cb.Long,
		RunE:  cb.RunE,
	}
}

// WrapError provides consistent error wrapping
func WrapError(operation string, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%s: %w", operation, err)
}
