package cmd

import (
	"context"
	"fmt"
	"time"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/constants"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/k3d"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/kubernetes"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
	"github.com/spf13/cobra"
)

func ClusterCmd() *cobra.Command {
	var clusterName string
	var kustomizePath string

	cmd := &cobra.Command{
		Use:   "cluster",
		Short: "Manage k3d cluster for local Kubernetes development",
		Long:  `Create, destroy, and manage k3d clusters with pre-configured infrastructure components.`,
	}

	cmd.PersistentFlags().StringVarP(&clusterName, "name", "n", constants.DefaultClusterName, "Cluster name")
	cmd.PersistentFlags().StringVarP(&kustomizePath, "kustomize-path", "k", "", "Path to kustomize overlay (optional)")

	cmd.AddCommand(clusterUpCmd(&clusterName, &kustomizePath))
	cmd.AddCommand(clusterDownCmd(&clusterName))
	cmd.AddCommand(clusterStatusCmd(&clusterName))
	cmd.AddCommand(clusterLogsCmd(&clusterName))
	cmd.AddCommand(clusterTestCmd(&clusterName))

	return cmd
}

func clusterUpCmd(clusterName *string, kustomizePath *string) *cobra.Command {
	return &cobra.Command{
		Use:   "up",
		Short: "Create k3d cluster and deploy infrastructure",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			logger := utils.NewLogger("cluster")

			// Create cluster
			cm := k3d.NewClusterManager(*clusterName)
			if err := cm.CreateCluster(ctx); err != nil {
				return fmt.Errorf("failed to create cluster: %w", err)
			}

			// Wait for cluster to be ready
			logger.Info("Waiting for cluster to be ready...")
			time.Sleep(10 * time.Second)

			// Create namespaces
			k8sClient, err := kubernetes.GetDefaultClient()
			if err != nil {
				return fmt.Errorf("failed to create k8s client: %w", err)
			}

			logger.Info("Creating namespaces...")
			namespaces := []string{constants.AppNamespace, constants.InfraNamespace}
			for _, ns := range namespaces {
				if err := k8sClient.CreateNamespace(ctx, ns); err != nil {
					logger.Warn("Failed to create namespace %s: %v", ns, err)
				}
			}

			// Deploy infrastructure
			logger.Info("Deploying infrastructure components...")

			// Find kustomize path
			var kustomizeDir string
			if *kustomizePath != "" {
				// Use the provided path directly
				kustomizeDir = *kustomizePath
			} else {
				var kErr error
				kustomizeDir, kErr = utils.GetKustomizePath("local")
				if kErr != nil {
					return fmt.Errorf("failed to get kustomize path: %w", kErr)
				}
			}

			if err := kubernetes.ApplyKustomize(ctx, kustomizeDir, ""); err != nil {
				return fmt.Errorf("failed to apply kustomize: %w", err)
			}

			// Wait for infrastructure to be ready
			logger.Info("Waiting for infrastructure components...")
			if err := waitForInfrastructure(ctx, k8sClient); err != nil {
				return fmt.Errorf("infrastructure failed to start: %w", err)
			}

			logger.Info("Cluster is ready!")
			logger.Info("Access the application at: http://localhost:8080")

			return nil
		},
	}
}

func clusterDownCmd(clusterName *string) *cobra.Command {
	return &cobra.Command{
		Use:   "down",
		Short: "Destroy k3d cluster and clean up resources",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			logger := utils.NewLogger("cluster")

			logger.Info("Destroying cluster %s...", *clusterName)

			cm := k3d.NewClusterManager(*clusterName)
			if err := cm.DeleteCluster(ctx); err != nil {
				return fmt.Errorf("failed to delete cluster: %w", err)
			}

			logger.Info("Cluster destroyed successfully")
			return nil
		},
	}
}

func clusterStatusCmd(clusterName *string) *cobra.Command {
	return &cobra.Command{
		Use:   "ps",
		Short: "Show cluster and resource status",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			logger := utils.NewLogger("cluster")

			// Check k3d cluster status
			cm := k3d.NewClusterManager(*clusterName)
			if !cm.IsRunning(ctx) {
				logger.Error("Cluster %s is not running", *clusterName)
				return nil
			}

			logger.Info("Cluster %s is running", *clusterName)

			// Show Kubernetes resources
			output, err := utils.RunCommand(ctx, "kubectl", []string{"get", "all", "--all-namespaces"}, nil)
			if err != nil {
				return fmt.Errorf("failed to get resources: %w", err)
			}

			fmt.Println(output)
			return nil
		},
	}
}

func clusterLogsCmd(_ *string) *cobra.Command {
	var namespace string
	var labelSelector string
	var tail int64

	cmd := &cobra.Command{
		Use:   "logs",
		Short: "Display pod logs",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()

			k8sClient, err := kubernetes.GetDefaultClient()
			if err != nil {
				return fmt.Errorf("failed to create k8s client: %w", err)
			}

			logRetriever := kubernetes.NewLogRetriever(k8sClient)
			opts := kubernetes.LogOptions{
				Namespace:     namespace,
				LabelSelector: labelSelector,
				TailLines:     tail,
				ShowPodName:   true,
			}

			logs, err := logRetriever.GetLogs(ctx, opts)
			if err != nil {
				return fmt.Errorf("failed to get logs: %w", err)
			}

			fmt.Print(logs)

			return nil
		},
	}

	cmd.Flags().StringVarP(&namespace, "namespace", "n", constants.AppNamespace, "Namespace")
	cmd.Flags().StringVarP(&labelSelector, "selector", "l", "", "Label selector")
	cmd.Flags().Int64Var(&tail, "tail", 50, "Number of lines to show")

	return cmd
}

func clusterTestCmd(clusterName *string) *cobra.Command {
	return &cobra.Command{
		Use:   "test",
		Short: "Verify deployment health",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			logger := utils.NewLogger("test")

			// Check cluster
			cm := k3d.NewClusterManager(*clusterName)
			if !cm.IsRunning(ctx) {
				return fmt.Errorf("cluster %s is not running", *clusterName)
			}
			logger.Info("✓ Cluster is running")

			// Check namespaces
			k8sClient, err := kubernetes.GetDefaultClient()
			if err != nil {
				return fmt.Errorf("failed to create k8s client: %w", err)
			}

			// Check infrastructure components
			checks := []struct {
				name      string
				namespace string
				selector  string
			}{
				{"PostgreSQL", constants.InfraNamespace, "app=postgres"},
				{"Redis", constants.InfraNamespace, "app=redis"},
			}

			for _, check := range checks {
				pods, err := k8sClient.GetPods(ctx, check.namespace, check.selector)
				if err != nil {
					logger.Error("✗ %s: %v", check.name, err)
					continue
				}

				running := 0
				for _, pod := range pods {
					if pod.Status.Phase == "Running" {
						running++
					}
				}

				if running > 0 {
					logger.Info("✓ %s: %d pod(s) running", check.name, running)
				} else {
					logger.Error("✗ %s: no pods running", check.name)
				}
			}

			return nil
		},
	}
}

func waitForInfrastructure(ctx context.Context, k8sClient *kubernetes.Client) error {
	logger := utils.NewLogger("wait")

	deployments := []struct {
		namespace string
		name      string
	}{
		{constants.InfraNamespace, "postgres"},
		{constants.InfraNamespace, "redis"},
	}

	for _, dep := range deployments {
		logger.Info("Waiting for %s/%s...", dep.namespace, dep.name)

		if err := k8sClient.WaitForDeployment(ctx, dep.namespace, dep.name, constants.DefaultTimeout); err != nil {
			logger.Error("Failed to wait for %s/%s: %v", dep.namespace, dep.name, err)
			// Continue anyway
		}
	}

	return nil
}
