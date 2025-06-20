package cmd

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/constants"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/kubernetes"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/secrets"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/testing"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
	"github.com/spf13/cobra"
)

func DebugCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "debug",
		Short: "Comprehensive debugging and management tool",
		Long:  `Debug deployments, manage secrets, and test API endpoints.`,
	}

	cmd.AddCommand(debugSecretsCmd())
	cmd.AddCommand(debugAnalyzeCmd())
	cmd.AddCommand(debugTestCmd())

	return cmd
}

func debugSecretsCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "secrets",
		Short: "Manage Kubernetes secrets",
	}

	cmd.AddCommand(debugSecretsCreateCmd())
	cmd.AddCommand(debugSecretsViewCmd())
	cmd.AddCommand(debugSecretsUpdateCmd())

	return cmd
}

func debugSecretsCreateCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "create",
		Short: "Create secrets from environment or .env file",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			logger := utils.NewLogger("secrets")

			// Load environment
			if err := utils.LoadEnvFile(); err != nil {
				logger.Warn("Failed to load .env file: %v", err)
			}

			// Create k8s client
			k8sClient, err := kubernetes.GetDefaultClient()
			if err != nil {
				return fmt.Errorf("failed to create k8s client: %w", err)
			}

			// Use secrets manager
			secretManager := secrets.NewManager(k8sClient)
			if err := secretManager.EnsureAppSecrets(ctx); err != nil {
				return fmt.Errorf("failed to create secrets: %w", err)
			}

			logger.Info("Secret created successfully")
			return nil
		},
	}
}

func debugSecretsViewCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "view",
		Short: "View current secrets",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			logger := utils.NewLogger("secrets")

			output, err := utils.RunCommand(ctx, "kubectl", []string{
				"get", "secret", "semantic-cache-secrets", "-n", constants.AppNamespace, "-o", "yaml",
			}, nil)

			if err != nil {
				logger.Error("Failed to get secret: %v", err)
				return err
			}

			// Decode the secret values for display (be careful with this in production!)
			fmt.Println("Current secrets (decoded):")
			fmt.Println(output)

			return nil
		},
	}
}

func debugSecretsUpdateCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "update",
		Short: "Update secrets from environment",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			logger := utils.NewLogger("secrets")

			// Load environment
			if err := utils.LoadEnvFile(); err != nil {
				logger.Warn("Failed to load .env file: %v", err)
			}

			// Create k8s client
			k8sClient, err := kubernetes.GetDefaultClient()
			if err != nil {
				return fmt.Errorf("failed to create k8s client: %w", err)
			}

			// Update secret
			apiKey := os.Getenv("OPENAI_API_KEY")
			dbURL := os.Getenv("DATABASE_URL")

			if apiKey == "" && dbURL == "" {
				return fmt.Errorf("no environment variables to update")
			}

			secretData := make(map[string][]byte)
			if apiKey != "" {
				secretData["openai-api-key"] = []byte(apiKey)
			}
			if dbURL != "" {
				secretData["database-url"] = []byte(dbURL)
			}

			if err := k8sClient.UpdateSecret(ctx, constants.AppNamespace, "semantic-cache-secrets", secretData); err != nil {
				return fmt.Errorf("failed to update secret: %w", err)
			}

			logger.Info("Secret updated successfully")
			return nil
		},
	}
}

func debugAnalyzeCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "analyze",
		Short: "Analyze deployment issues",
	}

	cmd.AddCommand(debugAnalyzeFullCmd())
	cmd.AddCommand(debugAnalyzeQuickCmd())

	return cmd
}

func debugAnalyzeFullCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "full",
		Short: "Full diagnostic analysis",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			logger := utils.NewLogger("analyze")

			logger.Info("Running full diagnostic analysis...")

			// Check cluster
			if err := analyzeCluster(ctx, logger); err != nil {
				logger.Error("Cluster analysis failed: %v", err)
			}

			// Check namespaces
			if err := analyzeNamespaces(ctx, logger); err != nil {
				logger.Error("Namespace analysis failed: %v", err)
			}

			// Check deployments
			if err := analyzeDeployments(ctx, logger); err != nil {
				logger.Error("Deployment analysis failed: %v", err)
			}

			// Check services
			if err := analyzeServices(ctx, logger); err != nil {
				logger.Error("Service analysis failed: %v", err)
			}

			// Check pods
			if err := analyzePods(ctx, logger); err != nil {
				logger.Error("Pod analysis failed: %v", err)
			}

			// Check secrets
			if err := analyzeSecrets(ctx, logger); err != nil {
				logger.Error("Secret analysis failed: %v", err)
			}

			return nil
		},
	}
}

func debugAnalyzeQuickCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "quick",
		Short: "Quick summary with recommendations",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			logger := utils.NewLogger("analyze")

			logger.Info("Running quick analysis...")

			// Check if cluster is running
			clusterRunning := checkClusterStatus(ctx)
			if !clusterRunning {
				logger.Error("Cluster is not running")
				logger.Info("Recommendation: Run 'iaac cluster up'")
				return nil
			}
			logger.Info("✓ Cluster is running")

			// Check if app is deployed
			appDeployed := checkAppDeployment(ctx)
			if !appDeployed {
				logger.Error("Application is not deployed")
				logger.Info("Recommendation: Run 'iaac dev deploy'")
				return nil
			}
			logger.Info("✓ Application is deployed")

			// Check if app is healthy
			appHealthy := checkAppHealth()
			if !appHealthy {
				logger.Error("Application is not healthy")
				logger.Info("Recommendation: Check logs with 'iaac dev logs'")
				return nil
			}
			logger.Info("✓ Application is healthy")

			logger.Info("Everything looks good!")
			return nil
		},
	}
}

func debugTestCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "test",
		Short: "Test API endpoints",
	}

	cmd.AddCommand(debugTestQuickCmd())
	cmd.AddCommand(debugTestDetailedCmd())

	return cmd
}

func debugTestQuickCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "quick",
		Short: "Quick API tests",
		RunE: func(cmd *cobra.Command, args []string) error {
			tester := testing.NewEndpointTester("http://localhost:8080")
			_, err := tester.TestStandardEndpoints()
			return err
		},
	}
}

func debugTestDetailedCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "detailed",
		Short: "Detailed endpoint analysis",
		RunE: func(cmd *cobra.Command, args []string) error {
			tester := testing.NewEndpointTester("http://localhost:8080")
			return tester.TestCacheOperations()
		},
	}
}

// Helper functions for analysis
func analyzeCluster(ctx context.Context, logger *utils.Logger) error {
	output, err := utils.RunCommand(ctx, "k3d", []string{"cluster", "list"}, nil)
	if err != nil {
		return err
	}
	logger.Info("Clusters:\n%s", output)
	return nil
}

func analyzeNamespaces(ctx context.Context, logger *utils.Logger) error {
	output, err := utils.RunCommand(ctx, "kubectl", []string{"get", "namespaces"}, nil)
	if err != nil {
		return err
	}
	logger.Info("Namespaces:\n%s", output)
	return nil
}

func analyzeDeployments(ctx context.Context, logger *utils.Logger) error {
	output, err := utils.RunCommand(ctx, "kubectl", []string{"get", "deployments", "--all-namespaces"}, nil)
	if err != nil {
		return err
	}
	logger.Info("Deployments:\n%s", output)
	return nil
}

func analyzeServices(ctx context.Context, logger *utils.Logger) error {
	output, err := utils.RunCommand(ctx, "kubectl", []string{"get", "services", "--all-namespaces"}, nil)
	if err != nil {
		return err
	}
	logger.Info("Services:\n%s", output)
	return nil
}

func analyzePods(ctx context.Context, logger *utils.Logger) error {
	output, err := utils.RunCommand(ctx, "kubectl", []string{"get", "pods", "--all-namespaces"}, nil)
	if err != nil {
		return err
	}
	logger.Info("Pods:\n%s", output)

	// Get problematic pods
	output2, err := utils.RunCommand(ctx, "kubectl", []string{
		"get", "pods", "--all-namespaces",
		"--field-selector=status.phase!=Running,status.phase!=Succeeded",
	}, nil)
	if err == nil && strings.TrimSpace(output2) != "" {
		logger.Warn("Problematic pods:\n%s", output2)
	}

	return nil
}

func analyzeSecrets(ctx context.Context, logger *utils.Logger) error {
	output, err := utils.RunCommand(ctx, "kubectl", []string{
		"get", "secrets", "-n", constants.AppNamespace,
	}, nil)
	if err != nil {
		return err
	}
	logger.Info("Secrets in app namespace:\n%s", output)
	return nil
}

func checkClusterStatus(ctx context.Context) bool {
	output, err := utils.RunCommand(ctx, "k3d", []string{"cluster", "list"}, nil)
	if err != nil {
		return false
	}
	return strings.Contains(output, "semantic-cache") && strings.Contains(output, "running")
}

func checkAppDeployment(ctx context.Context) bool {
	output, err := utils.RunCommand(ctx, "kubectl", []string{
		"get", "deployment", "semantic-cache", "-n", constants.AppNamespace,
	}, nil)
	return err == nil && strings.Contains(output, "semantic-cache")
}

func checkAppHealth() bool {
	resp, err := http.Get("http://localhost:8080/health")
	if err != nil {
		return false
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			// Log error if we have a way to do so, otherwise ignore
			// since this is just a health check
			_ = closeErr // Acknowledge the error to satisfy linter
		}
	}()
	return resp.StatusCode == 200
}
