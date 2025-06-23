// Package commands provides Docker-related commands
package commands

import (
	"context"
	"fmt"

	"github.com/spf13/cobra"

	"github.com/raja-aiml/sematic-cache/devops/pkg/factory"
)

// DockerCommand handles Docker-related operations
type DockerCommand struct {
	*BaseCommand
	cmd *cobra.Command
}

// NewDockerCommand creates a new docker command group
func NewDockerCommand(factory *factory.Factory) *DockerCommand {
	dc := &DockerCommand{
		BaseCommand: NewBaseCommand(factory),
	}

	dc.cmd = &cobra.Command{
		Use:   "docker",
		Short: "Docker operations",
		Long:  "Perform various Docker operations using the Docker SDK",
	}

	// Add subcommands
	dc.cmd.AddCommand(dc.newStatusCommand())
	dc.cmd.AddCommand(dc.newListCommand())

	return dc
}

// GetCommand returns the cobra command
func (dc *DockerCommand) GetCommand() *cobra.Command {
	return dc.cmd
}

// newStatusCommand creates the docker status subcommand
func (dc *DockerCommand) newStatusCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "status",
		Short: "Check Docker daemon status",
		RunE: func(cmd *cobra.Command, args []string) error {
			return dc.RunWithContext(func(ctx context.Context) error {
				client, err := dc.GetDockerClient()
				if err != nil {
					return fmt.Errorf("failed to create Docker client: %w", err)
				}

				if client.IsRunning(ctx) {
					dc.logger.Success("Docker daemon is running")
				} else {
					dc.logger.Error("Docker daemon is not running")
					return fmt.Errorf("Docker daemon is not accessible")
				}

				return nil
			})
		},
	}
}

// newListCommand creates the docker list subcommand
func (dc *DockerCommand) newListCommand() *cobra.Command {
	var all bool

	cmd := &cobra.Command{
		Use:   "list",
		Short: "List Docker containers",
		RunE: func(cmd *cobra.Command, args []string) error {
			return dc.RunWithContext(func(ctx context.Context) error {
				client, err := dc.GetDockerClient()
				if err != nil {
					return fmt.Errorf("failed to create Docker client: %w", err)
				}

				containers, err := client.ListContainers(ctx, all)
				if err != nil {
					return fmt.Errorf("failed to list containers: %w", err)
				}

				if len(containers) == 0 {
					dc.logger.Info("No containers found")
					return nil
				}

				dc.logger.Info("Found %d containers:", len(containers))
				for _, container := range containers {
					fmt.Printf("  %s: %s (%s) - %s\n",
						container.ID,
						container.Name,
						container.Image,
						container.Status)
				}

				return nil
			})
		},
	}

	cmd.Flags().BoolVarP(&all, "all", "a", false, "Show all containers (default shows just running)")

	return cmd
}
