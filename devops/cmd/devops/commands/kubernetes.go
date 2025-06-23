// Package commands provides Kubernetes-related commands
package commands

import (
	"context"
	"fmt"

	"github.com/spf13/cobra"

	"github.com/raja-aiml/sematic-cache/devops/pkg/factory"
)

// KubernetesCommand handles Kubernetes-related operations
type KubernetesCommand struct {
	*BaseCommand
	cmd *cobra.Command
}

// NewKubernetesCommand creates a new kubernetes command group
func NewKubernetesCommand(factory *factory.Factory) *KubernetesCommand {
	kc := &KubernetesCommand{
		BaseCommand: NewBaseCommand(factory),
	}

	kc.cmd = &cobra.Command{
		Use:     "k8s",
		Aliases: []string{"kubernetes"},
		Short:   "Kubernetes operations",
		Long:    "Perform various Kubernetes operations using the client-go SDK",
	}

	// Add subcommands
	kc.cmd.AddCommand(kc.newContextCommand())
	kc.cmd.AddCommand(kc.newPodsCommand())
	kc.cmd.AddCommand(kc.newServicesCommand())

	return kc
}

// GetCommand returns the cobra command
func (kc *KubernetesCommand) GetCommand() *cobra.Command {
	return kc.cmd
}

// newContextCommand creates the k8s context subcommand
func (kc *KubernetesCommand) newContextCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "context",
		Short: "Show current Kubernetes context",
		RunE: func(cmd *cobra.Command, args []string) error {
			return kc.RunWithContext(func(ctx context.Context) error {
				client, err := kc.GetKubernetesClient()
				if err != nil {
					return fmt.Errorf("failed to create Kubernetes client: %w", err)
				}

				context, err := client.GetCurrentContext()
				if err != nil {
					return fmt.Errorf("failed to get current context: %w", err)
				}

				if context == "" {
					kc.logger.Warning("No Kubernetes context configured")
				} else {
					kc.logger.Success("Current Kubernetes context: %s", context)
				}

				return nil
			})
		},
	}
}

// newPodsCommand creates the k8s pods subcommand
func (kc *KubernetesCommand) newPodsCommand() *cobra.Command {
	var namespace string
	var labelSelector string

	cmd := &cobra.Command{
		Use:   "pods",
		Short: "List Kubernetes pods",
		RunE: func(cmd *cobra.Command, args []string) error {
			return kc.RunWithContext(func(ctx context.Context) error {
				client, err := kc.GetKubernetesClient()
				if err != nil {
					return fmt.Errorf("failed to create Kubernetes client: %w", err)
				}

				pods, err := client.GetPods(ctx, namespace, labelSelector)
				if err != nil {
					return fmt.Errorf("failed to list pods: %w", err)
				}

				if len(pods) == 0 {
					kc.logger.Info("No pods found")
					return nil
				}

				kc.logger.Info("Found %d pods:", len(pods))
				for _, pod := range pods {
					status := "Not Ready"
					if pod.Ready {
						status = "Ready"
					}
					fmt.Printf("  %s/%s: %s (%s) on %s\n",
						pod.Namespace,
						pod.Name,
						status,
						pod.Status,
						pod.Node)
				}

				return nil
			})
		},
	}

	cmd.Flags().StringVarP(&namespace, "namespace", "n", "default", "Kubernetes namespace")
	cmd.Flags().StringVarP(&labelSelector, "selector", "l", "", "Label selector")

	return cmd
}

// newServicesCommand creates the k8s services subcommand
func (kc *KubernetesCommand) newServicesCommand() *cobra.Command {
	var namespace string

	cmd := &cobra.Command{
		Use:   "services",
		Short: "List Kubernetes services",
		RunE: func(cmd *cobra.Command, args []string) error {
			return kc.RunWithContext(func(ctx context.Context) error {
				client, err := kc.GetKubernetesClient()
				if err != nil {
					return fmt.Errorf("failed to create Kubernetes client: %w", err)
				}

				services, err := client.GetServices(ctx, namespace)
				if err != nil {
					return fmt.Errorf("failed to list services: %w", err)
				}

				if len(services) == 0 {
					kc.logger.Info("No services found")
					return nil
				}

				kc.logger.Info("Found %d services:", len(services))
				for _, svc := range services {
					fmt.Printf("  %s/%s: %s (%s)\n",
						svc.Namespace,
						svc.Name,
						svc.Type,
						svc.ClusterIP)

					for _, port := range svc.Ports {
						fmt.Printf("    Port: %s %d -> %d\n",
							port.Protocol,
							port.Port,
							port.TargetPort)
					}
				}

				return nil
			})
		},
	}

	cmd.Flags().StringVarP(&namespace, "namespace", "n", "default", "Kubernetes namespace")

	return cmd
}
