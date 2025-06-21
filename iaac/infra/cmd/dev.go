package cmd

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/config"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/constants"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/docker"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/kubernetes"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/secrets"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/testing"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
	"github.com/spf13/cobra"
)

func DevCmd() *cobra.Command {
	var imageName string
	var clusterName string
	var kustomizePath string

	cmd := &cobra.Command{
		Use:   "dev",
		Short: "Manage application build and deployment",
		Long:  `Build Docker images, deploy applications, and manage the development lifecycle.`,
	}

	cmd.PersistentFlags().StringVarP(&imageName, "image", "i", constants.DefaultImageName, "Docker image name")
	cmd.PersistentFlags().StringVarP(&clusterName, "cluster", "c", constants.DefaultClusterName, "k3d cluster name")
	cmd.PersistentFlags().StringVarP(&kustomizePath, "kustomize-path", "k", "", "Path to kustomize overlay (optional)")

	cmd.AddCommand(devBuildCmd(&imageName, &clusterName))
	cmd.AddCommand(devDeployCmd(&kustomizePath))
	cmd.AddCommand(devTestCmd())
	cmd.AddCommand(devRemoveCmd(&kustomizePath))
	cmd.AddCommand(devLogsCmd())
	cmd.AddCommand(devStatusCmd())

	return cmd
}

func devBuildCmd(imageName, clusterName *string) *cobra.Command {
	return &cobra.Command{
		Use:   "build",
		Short: "Build Docker image and import to k3d",
		RunE: func(cmd *cobra.Command, args []string) error {
			// Load configuration
			if err := config.LoadConfig(cmd); err != nil {
				return fmt.Errorf("failed to load config: %w", err)
			}

			ctx := context.Background()
			logger := utils.NewLogger("build")

			// Find project root
			projectRoot, err := utils.FindProjectRoot()
			if err != nil {
				return fmt.Errorf("failed to find project root: %w", err)
			}

			// Build image
			builder, err := docker.NewBuilder()
			if err != nil {
				return fmt.Errorf("failed to create Docker builder: %w", err)
			}
			defer builder.Close()

			dockerfilePath := filepath.Join(projectRoot, "Dockerfile")

			if err := builder.BuildSimple(ctx, dockerfilePath, *imageName, projectRoot); err != nil {
				return fmt.Errorf("failed to build image: %w", err)
			}

			// Test image locally
			logger.Info("Testing image locally...")
			config := &docker.ContainerConfig{
				Cmd:        []string{"--help"},
				AutoRemove: true,
			}
			if _, err := builder.RunContainer(ctx, *imageName, config); err != nil {
				logger.Warn("Local test failed: %v", err)
			}

			// Import to k3d
			if err := builder.ImportToK3d(ctx, *imageName, *clusterName); err != nil {
				return fmt.Errorf("failed to import image to k3d: %w", err)
			}

			logger.Info("Build and import completed successfully")
			return nil
		},
	}
}

func devDeployCmd(kustomizePath *string) *cobra.Command {
	return &cobra.Command{
		Use:   "deploy",
		Short: "Create secrets and deploy application",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			logger := utils.NewLogger("deploy")

			// Load environment variables
			if err := utils.LoadEnvFile(); err != nil {
				logger.Warn("Failed to load .env file: %v", err)
			}

			// Create k8s client
			k8sClient, err := kubernetes.GetDefaultClient()
			if err != nil {
				return fmt.Errorf("failed to create k8s client: %w", err)
			}

			// Create/update secrets
			logger.Info("Creating secrets...")
			secretManager := secrets.NewManager(k8sClient)
			if err := secretManager.EnsureAppSecrets(ctx); err != nil {
				return fmt.Errorf("failed to create secrets: %w", err)
			}

			// Deploy application
			logger.Info("Deploying application...")
			var deployPath string
			if *kustomizePath != "" {
				// Use the provided path directly
				deployPath = *kustomizePath
			} else {
				deployPath, err = utils.GetKustomizePath("local/app")
				if err != nil {
					return fmt.Errorf("failed to get kustomize path: %w", err)
				}
			}

			if err := kubernetes.ApplyKustomize(ctx, deployPath, constants.AppNamespace); err != nil {
				return fmt.Errorf("failed to deploy application: %w", err)
			}

			// Wait for deployment
			logger.Info("Waiting for application to be ready...")
			if err := k8sClient.WaitForDeployment(ctx, constants.AppNamespace, "semantic-cache", constants.DefaultTimeout); err != nil {
				return fmt.Errorf("application failed to start: %w", err)
			}

			logger.Info("Application deployed successfully")
			logger.Info("Access the application at: http://localhost:8080")

			return nil
		},
	}
}

func devTestCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "test",
		Short: "Test application endpoints",
		RunE: func(cmd *cobra.Command, args []string) error {
			tester := testing.NewEndpointTester("http://localhost:8080")

			results, err := tester.TestStandardEndpoints()
			if err != nil {
				return fmt.Errorf("endpoint tests failed: %w", err)
			}

			allPassed := true
			for _, result := range results {
				if !result.Success {
					allPassed = false
				}
			}

			if !allPassed {
				return fmt.Errorf("some tests failed")
			}

			logger := utils.NewLogger("test")
			logger.Info("All tests passed!")
			return nil
		},
	}
}

func devRemoveCmd(kustomizePath *string) *cobra.Command {
	return &cobra.Command{
		Use:   "remove",
		Short: "Remove application deployment",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			logger := utils.NewLogger("remove")

			var removePath string
			var err error
			if *kustomizePath != "" {
				// Use the provided path directly
				removePath = *kustomizePath
			} else {
				removePath, err = utils.GetKustomizePath("local/app")
				if err != nil {
					return fmt.Errorf("failed to get kustomize path: %w", err)
				}
			}

			logger.Info("Removing application...")
			if err := kubernetes.DeleteKustomize(ctx, removePath, constants.AppNamespace); err != nil {
				logger.Warn("Remove completed with warnings: %v", err)
			}

			logger.Info("Application removed")
			return nil
		},
	}
}

func devLogsCmd() *cobra.Command {
	var follow bool
	var tail int64

	cmd := &cobra.Command{
		Use:   "logs",
		Short: "Show application logs",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()

			k8sClient, err := kubernetes.GetDefaultClient()
			if err != nil {
				return fmt.Errorf("failed to create k8s client: %w", err)
			}

			logRetriever := kubernetes.NewLogRetriever(k8sClient)
			opts := kubernetes.LogOptions{
				Namespace:     constants.AppNamespace,
				LabelSelector: "app=semantic-cache",
				TailLines:     tail,
				Follow:        follow,
				ShowPodName:   true,
			}

			return logRetriever.StreamLogs(ctx, opts)
		},
	}

	cmd.Flags().BoolVarP(&follow, "follow", "f", false, "Follow log output")
	cmd.Flags().Int64Var(&tail, "tail", 100, "Number of lines to show")

	return cmd
}

func devStatusCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "status",
		Short: "Display deployment status",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			logger := utils.NewLogger("status")

			// Show deployment status
			output, err := utils.RunCommand(ctx, "kubectl", []string{
				"get", "all", "-n", constants.AppNamespace, "-l", "app=semantic-cache",
			}, nil)

			if err != nil {
				logger.Error("Failed to get status: %v", err)
				return err
			}

			fmt.Println(output)

			// Check if app is accessible
			tester := testing.NewEndpointTester("http://localhost:8080")
			result := tester.TestEndpoint("GET", "/health", nil)

			if result.Success {
				logger.Info("Application is healthy and accessible at http://localhost:8080")
			} else if result.Error != nil {
				logger.Warn("Application not accessible at http://localhost:8080: %v", result.Error)
			} else {
				logger.Warn("Application returned status %d", result.StatusCode)
			}

			return nil
		},
	}
}
