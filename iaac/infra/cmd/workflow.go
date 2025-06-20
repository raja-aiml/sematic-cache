package cmd

import (
	"context"
	"fmt"
	"os"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/constants"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
	"github.com/spf13/cobra"
)

type WorkflowManager struct {
	logger      *utils.Logger
	clusterName string
	imageName   string
	scenario    string
	overlay     string
}

func WorkflowCmd() *cobra.Command {
	var clusterName string
	var imageName string
	var scenario string
	var overlay string

	cmd := &cobra.Command{
		Use:   "workflow",
		Short: "Production-ready end-to-end workflow orchestrator",
		Long:  `Manage complete deployment lifecycle including cluster setup, build, deploy, test, and cleanup.`,
	}

	cmd.PersistentFlags().StringVarP(&clusterName, "cluster", "c", constants.DefaultClusterName, "k3d cluster name")
	cmd.PersistentFlags().StringVarP(&imageName, "image", "i", constants.DefaultImageName, "Docker image name")
	cmd.PersistentFlags().StringVarP(&scenario, "scenario", "s", constants.ScenarioDevelopment, "Blueprint scenario to deploy")
	cmd.PersistentFlags().StringVarP(&overlay, "overlay", "o", "local", "Overlay to use (local, dev)")

	wm := &WorkflowManager{
		logger:      utils.NewLogger("workflow"),
		clusterName: clusterName,
		imageName:   imageName,
		scenario:    scenario,
		overlay:     overlay,
	}

	cmd.AddCommand(workflowFullCmd(wm))
	cmd.AddCommand(workflowSetupCmd(wm))
	cmd.AddCommand(workflowBuildCmd(wm))
	cmd.AddCommand(workflowDeployCmd(wm))
	cmd.AddCommand(workflowTestCmd(wm))
	cmd.AddCommand(workflowCleanupCmd(wm))
	cmd.AddCommand(workflowStatusCmd(wm))
	cmd.AddCommand(workflowLogsCmd(wm))
	cmd.AddCommand(workflowResetCmd(wm))

	return cmd
}

func workflowFullCmd(wm *WorkflowManager) *cobra.Command {
	return &cobra.Command{
		Use:   "full",
		Short: "Run complete workflow (setup, build, deploy, test)",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			wm.logger.Info("Starting full workflow...")

			// Setup
			wm.logger.Step("Phase 1/4: Setup")
			if err := runSetup(ctx, wm); err != nil {
				return fmt.Errorf("setup failed: %w", err)
			}

			// Build
			wm.logger.Step("Phase 2/4: Build")
			if err := runBuild(ctx, wm); err != nil {
				return fmt.Errorf("build failed: %w", err)
			}

			// Deploy
			wm.logger.Step("Phase 3/4: Deploy")
			if err := runDeploy(ctx, wm); err != nil {
				return fmt.Errorf("deploy failed: %w", err)
			}

			// Test
			wm.logger.Step("Phase 4/4: Test")
			if err := runTest(ctx, wm); err != nil {
				return fmt.Errorf("test failed: %w", err)
			}

			wm.logger.Info("Workflow completed successfully!")
			if wm.scenario == constants.ScenarioMinimal {
				wm.logger.Info("Infrastructure components are ready!")
			} else {
				wm.logger.Info("Access the application at: http://localhost:8080")
			}

			return nil
		},
	}
}

func workflowSetupCmd(wm *WorkflowManager) *cobra.Command {
	return &cobra.Command{
		Use:   "setup",
		Short: "Create cluster and deploy infrastructure",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			return runSetup(ctx, wm)
		},
	}
}

func workflowBuildCmd(wm *WorkflowManager) *cobra.Command {
	return &cobra.Command{
		Use:   "build",
		Short: "Build and import Docker image",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			return runBuild(ctx, wm)
		},
	}
}

func workflowDeployCmd(wm *WorkflowManager) *cobra.Command {
	return &cobra.Command{
		Use:   "deploy",
		Short: "Deploy application with secrets",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			return runDeploy(ctx, wm)
		},
	}
}

func workflowTestCmd(wm *WorkflowManager) *cobra.Command {
	return &cobra.Command{
		Use:   "test",
		Short: "Run comprehensive e2e tests",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			return runTest(ctx, wm)
		},
	}
}

func workflowCleanupCmd(wm *WorkflowManager) *cobra.Command {
	return &cobra.Command{
		Use:   "cleanup",
		Short: "Remove application deployment",
		RunE: func(cmd *cobra.Command, args []string) error {
			wm.logger.Info("Cleaning up application...")

			// Remove application
			kustomizePath := ""
			devRemove := devRemoveCmd(&kustomizePath)
			if err := devRemove.RunE(devRemove, []string{}); err != nil {
				wm.logger.Warn("Cleanup completed with warnings: %v", err)
			}

			wm.logger.Info("Cleanup completed")
			return nil
		},
	}
}

func workflowStatusCmd(wm *WorkflowManager) *cobra.Command {
	return &cobra.Command{
		Use:   "status",
		Short: "Show workflow status",
		RunE: func(cmd *cobra.Command, args []string) error {
			wm.logger.Info("Checking workflow status...")

			// Check cluster
			clusterStatus := clusterStatusCmd(&wm.clusterName)
			if err := clusterStatus.RunE(clusterStatus, []string{}); err != nil {
				wm.logger.Error("Cluster status check failed: %v", err)
			}

			// Check application
			devStatus := devStatusCmd()
			if err := devStatus.RunE(devStatus, []string{}); err != nil {
				wm.logger.Warn("Application status check failed: %v", err)
			}

			return nil
		},
	}
}

func workflowLogsCmd(_ *WorkflowManager) *cobra.Command {
	return &cobra.Command{
		Use:   "logs",
		Short: "Show application and infrastructure logs",
		RunE: func(cmd *cobra.Command, args []string) error {
			// Show application logs
			devLogs := devLogsCmd()
			return devLogs.RunE(devLogs, []string{})
		},
	}
}

func workflowResetCmd(wm *WorkflowManager) *cobra.Command {
	return &cobra.Command{
		Use:   "reset",
		Short: "Reset everything (destroy cluster and clean up)",
		RunE: func(cmd *cobra.Command, args []string) error {
			wm.logger.Warn("Resetting environment...")

			// Destroy cluster
			clusterDown := clusterDownCmd(&wm.clusterName)
			if err := clusterDown.RunE(clusterDown, []string{}); err != nil {
				wm.logger.Error("Failed to destroy cluster: %v", err)
			}

			wm.logger.Info("Reset completed")
			return nil
		},
	}
}

// Helper functions for workflow steps
func runSetup(_ context.Context, wm *WorkflowManager) error {
	wm.logger.Info("Setting up cluster and infrastructure...")

	// Create cluster
	clusterName := wm.clusterName
	scenario := wm.scenario
	overlay := wm.overlay
	kustomizePath := ""
	clusterUp := clusterUpCmd(&clusterName, &scenario, &overlay, &kustomizePath)
	if err := clusterUp.RunE(clusterUp, []string{}); err != nil {
		return fmt.Errorf("cluster setup failed: %w", err)
	}

	return nil
}

func runBuild(_ context.Context, wm *WorkflowManager) error {
	wm.logger.Info("Building application...")

	// Build and import image
	devBuild := devBuildCmd(&wm.imageName, &wm.clusterName)
	if err := devBuild.RunE(devBuild, []string{}); err != nil {
		return fmt.Errorf("build failed: %w", err)
	}

	return nil
}

func runDeploy(_ context.Context, wm *WorkflowManager) error {
	wm.logger.Info("Deploying application...")

	// Deploy application
	kPath := ""
	devDeploy := devDeployCmd(&kPath)
	if err := devDeploy.RunE(devDeploy, []string{}); err != nil {
		return fmt.Errorf("deploy failed: %w", err)
	}

	return nil
}

func runTest(ctx context.Context, wm *WorkflowManager) error {
	wm.logger.Info("Running tests...")

	// Run basic endpoint tests
	devTest := devTestCmd()
	if err := devTest.RunE(devTest, []string{}); err != nil {
		return fmt.Errorf("tests failed: %w", err)
	}

	// Run e2e tests if available
	projectRoot, err := utils.FindProjectRoot()
	if err == nil {
		e2eScript := fmt.Sprintf("%s/deploy/tests/e2e.sh", projectRoot)
		if _, err := os.Stat(e2eScript); err == nil {
			wm.logger.Info("Running e2e tests...")
			output, err := utils.RunShellCommand(ctx, e2eScript+" quick", nil)
			if err != nil {
				wm.logger.Warn("E2E tests failed: %v", err)
			} else {
				wm.logger.Debug("E2E test output: %s", output)
			}
		}
	}

	wm.logger.Info("All tests completed")
	return nil
}
