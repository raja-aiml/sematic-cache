package cmd

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/blueprint"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/config"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/constants"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/k3d"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/kubernetes"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
	"github.com/spf13/cobra"
)

func ClusterCmd() *cobra.Command {
	var clusterName string
	var scenario string
	var overlay string
	var kustomizePath string

	cmd := &cobra.Command{
		Use:   "cluster",
		Short: "Manage k3d cluster with blueprint scenarios",
		Long:  `Create, destroy, and manage k3d clusters using pre-configured blueprint scenarios.`,
	}

	cmd.PersistentFlags().StringVarP(&clusterName, "name", "n", constants.DefaultClusterName, "Cluster name")
	cmd.PersistentFlags().StringVarP(&scenario, "scenario", "s", "minimal", "Blueprint scenario to deploy")
	cmd.PersistentFlags().StringVarP(&overlay, "overlay", "o", "local", "Overlay to use (local, dev)")
	cmd.PersistentFlags().StringVarP(&kustomizePath, "kustomize-path", "k", "", "Path to kustomize overlay (overrides scenario)")

	cmd.AddCommand(clusterUpCmd(&clusterName, &scenario, &overlay, &kustomizePath))
	cmd.AddCommand(clusterDownCmd(&clusterName))
	cmd.AddCommand(clusterStatusCmd(&clusterName))
	cmd.AddCommand(clusterLogsCmd(&clusterName))
	cmd.AddCommand(clusterTestCmd(&clusterName, &scenario))

	return cmd
}

func clusterUpCmd(clusterName *string, scenario *string, overlay *string, kustomizePath *string) *cobra.Command {
	return &cobra.Command{
		Use:   "up",
		Short: "Create k3d cluster and deploy blueprint scenario",
		Long: `Create a k3d cluster and deploy infrastructure using blueprint scenarios.
		
Available scenarios:
  - minimal: Basic PostgreSQL and Redis
  - development: Full development stack with debug tools
  - service-mesh: Istio service mesh with observability
  - monitoring-only: Just the observability stack
  - full-stack: Complete production-like environment`,
		RunE: func(cmd *cobra.Command, args []string) error {
			// Load configuration
			if err := config.LoadConfig(cmd); err != nil {
				return fmt.Errorf("failed to load config: %w", err)
			}

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

			// Get k8s client
			k8sClient, err := kubernetes.GetDefaultClient()
			if err != nil {
				return fmt.Errorf("failed to create k8s client: %w", err)
			}

			// Determine what to deploy
			var deployPath string
			if *kustomizePath != "" {
				// Use custom kustomize path if provided
				deployPath = *kustomizePath
				logger.Info("Using custom kustomize path: %s", deployPath)
			} else {
				// Use blueprint scenario
				workDir, err := os.Getwd()
				if err != nil {
					return fmt.Errorf("failed to get working directory: %w", err)
				}

				iaacPath := findIaacPath(workDir)
				if iaacPath == "" {
					return fmt.Errorf("could not find iaac directory")
				}

				// Build path based on scenario and overlay
				if *overlay != "" && *overlay != "base" {
					// Use overlay-specific path if provided
					overlayPath := filepath.Join(iaacPath, blueprint.GetOverlayPath(*overlay))
					if _, err := os.Stat(overlayPath); err == nil {
						deployPath = overlayPath
						logger.Info("Deploying with overlay: %s", *overlay)
					} else {
						// Fall back to scenario if overlay doesn't exist
						logger.Warn("Overlay '%s' not found, using scenario '%s' instead", *overlay, *scenario)
						deployPath = filepath.Join(iaacPath, blueprint.GetScenarioPath(*scenario))
					}
				} else {
					// Use scenario path
					deployPath = filepath.Join(iaacPath, blueprint.GetScenarioPath(*scenario))
					logger.Info("Deploying blueprint scenario: %s", *scenario)
				}
			}

			// Check if path exists
			if _, err := os.Stat(deployPath); os.IsNotExist(err) {
				return fmt.Errorf("deployment path not found: %s", deployPath)
			}

			// Apply the configuration
			if err := kubernetes.ApplyKustomize(ctx, deployPath, ""); err != nil {
				return fmt.Errorf("failed to apply configuration: %w", err)
			}

			// Wait for components based on scenario
			logger.Info("Waiting for components to be ready...")
			if err := waitForScenarioComponents(ctx, k8sClient, *scenario); err != nil {
				return fmt.Errorf("components failed to start: %w", err)
			}

			logger.Info("Deployment completed successfully!")
			printScenarioAccess(*scenario)

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
		Short: "Show cluster and blueprint component status",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			logger := utils.NewLogger("status")

			cm := k3d.NewClusterManager(*clusterName)
			if !cm.IsRunning(ctx) {
				logger.Error("Cluster %s is not running", *clusterName)
				return fmt.Errorf("cluster not running")
			}

			logger.Info("✓ Cluster %s is running", *clusterName)

			// Check components
			k8sClient, err := kubernetes.GetDefaultClient()
			if err != nil {
				return fmt.Errorf("failed to create k8s client: %w", err)
			}

			// Check all possible namespaces
			namespaces := []string{
				constants.InfraNamespace,
				constants.AppNamespace,
				constants.MonitoringNamespace,
				constants.IstioNamespace,
				constants.LoggingNamespace,
				constants.TracingNamespace,
			}

			logger.Info("\nNamespace Status:")
			for _, ns := range namespaces {
				exists, _ := k8sClient.NamespaceExists(ctx, ns)
				if exists {
					logger.Info("✓ %s", ns)

					// Check deployments in namespace
					deployments, err := k8sClient.GetDeployments(ctx, ns)
					if err == nil && len(deployments) > 0 {
						for _, dep := range deployments {
							if dep.Status.ReadyReplicas > 0 {
								logger.Info("  ✓ %s (%d/%d replicas ready)",
									dep.Name, dep.Status.ReadyReplicas, dep.Status.Replicas)
							} else {
								logger.Warn("  ✗ %s (0/%d replicas ready)",
									dep.Name, dep.Status.Replicas)
							}
						}
					}

					// Check statefulsets
					statefulsets, err := k8sClient.GetStatefulSets(ctx, ns)
					if err == nil && len(statefulsets) > 0 {
						for _, sts := range statefulsets {
							if sts.Status.ReadyReplicas > 0 {
								logger.Info("  ✓ %s (%d/%d replicas ready) [StatefulSet]",
									sts.Name, sts.Status.ReadyReplicas, sts.Status.Replicas)
							} else {
								logger.Warn("  ✗ %s (0/%d replicas ready) [StatefulSet]",
									sts.Name, sts.Status.Replicas)
							}
						}
					}
				}
			}

			// Show quick access commands
			logger.Info("\nQuick Access Commands:")
			logger.Info("  kubectl get pods --all-namespaces")
			logger.Info("  kubectl get svc --all-namespaces")
			logger.Info("  kubectl get ingress --all-namespaces")

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

func clusterTestCmd(clusterName *string, scenario *string) *cobra.Command {
	return &cobra.Command{
		Use:   "test",
		Short: "Run tests for the deployed blueprint scenario",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			logger := utils.NewLogger("test")

			// Check cluster
			cm := k3d.NewClusterManager(*clusterName)
			if !cm.IsRunning(ctx) {
				return fmt.Errorf("cluster %s is not running", *clusterName)
			}
			logger.Info("✓ Cluster is running")

			// Get k8s client
			k8sClient, err := kubernetes.GetDefaultClient()
			if err != nil {
				return fmt.Errorf("failed to create k8s client: %w", err)
			}

			// Run scenario-specific tests
			logger.Info("Running tests for scenario: %s", *scenario)

			switch *scenario {
			case "minimal":
				// Test basic infrastructure
				logger.Info("Testing minimal infrastructure components...")
				if err := runConnectivityTests(ctx, k8sClient); err != nil {
					logger.Error("Basic connectivity tests failed: %v", err)
				}

			case "development":
				// Test development stack
				logger.Info("Testing development stack...")
				if err := runConnectivityTests(ctx, k8sClient); err != nil {
					logger.Error("Basic connectivity tests failed: %v", err)
				}
				if err := runMonitoringTests(ctx, k8sClient); err != nil {
					logger.Error("Monitoring tests failed: %v", err)
				}

			case "service-mesh":
				// Test service mesh
				logger.Info("Testing service mesh components...")
				if err := runServiceMeshTests(ctx, k8sClient); err != nil {
					logger.Error("Service mesh tests failed: %v", err)
				}

			case "monitoring-only":
				// Test monitoring stack
				logger.Info("Testing monitoring stack...")
				if err := runMonitoringTests(ctx, k8sClient); err != nil {
					logger.Error("Monitoring tests failed: %v", err)
				}

			case "full-stack":
				// Test everything
				logger.Info("Testing full stack...")
				if err := runConnectivityTests(ctx, k8sClient); err != nil {
					logger.Error("Basic connectivity tests failed: %v", err)
				}
				if err := runServiceMeshTests(ctx, k8sClient); err != nil {
					logger.Error("Service mesh tests failed: %v", err)
				}
				if err := runMonitoringTests(ctx, k8sClient); err != nil {
					logger.Error("Monitoring tests failed: %v", err)
				}

			default:
				// Run basic tests for unknown scenarios
				logger.Info("Running basic tests...")
				if err := runConnectivityTests(ctx, k8sClient); err != nil {
					logger.Error("Connectivity tests failed: %v", err)
				}
			}

			// Get iaac path for validation scripts
			// Run smoke tests using Go implementation
			logger.Info("Running blueprint validation tests...")
			testCmd := TestCmd()
			testCmd.SetArgs([]string{"smoke", "--scenario", *scenario})
			if err := testCmd.Execute(); err != nil {
				logger.Error("Validation tests failed: %v", err)
			} else {
				logger.Info("✓ Validation tests passed")
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

// Helper functions for blueprint integration

func findIaacPath(startPath string) string {
	current := startPath
	for {
		iaacPath := filepath.Join(current, "iaac")
		if _, err := os.Stat(iaacPath); err == nil {
			return iaacPath
		}

		parent := filepath.Dir(current)
		if parent == current {
			break
		}
		current = parent
	}
	return ""
}

func waitForScenarioComponents(ctx context.Context, client *kubernetes.Client, scenario string) error {
	if ctx == nil || client == nil {
		return fmt.Errorf("invalid parameters: context and client must not be nil")
	}

	timeout := time.Duration(constants.DeploymentReadyTimeout) * time.Second

	switch scenario {
	case "minimal":
		return waitForMinimalComponents(ctx, client, timeout)
	case "development":
		return waitForDevelopmentComponents(ctx, client, timeout)
	case "service-mesh":
		return waitForServiceMeshComponents(ctx, client, timeout)
	case "monitoring-only":
		return waitForMonitoringComponents(ctx, client, timeout)
	case "full-stack":
		return waitForFullStackComponents(ctx, client, timeout)
	default:
		// For unknown scenarios or custom paths, wait for basic infrastructure
		return waitForInfrastructure(ctx, client)
	}
}

func waitForMinimalComponents(ctx context.Context, client *kubernetes.Client, timeout time.Duration) error {
	components := []struct {
		namespace string
		name      string
	}{
		{constants.InfraNamespace, "postgres"},
		{constants.InfraNamespace, "redis"},
	}

	return waitForDeployments(ctx, client, components, timeout)
}

func waitForDevelopmentComponents(ctx context.Context, client *kubernetes.Client, timeout time.Duration) error {
	// Wait for minimal components first
	if err := waitForMinimalComponents(ctx, client, timeout); err != nil {
		return err
	}

	// Additional development components
	components := []struct {
		namespace string
		name      string
	}{
		{constants.MonitoringNamespace, "prometheus"},
		{constants.MonitoringNamespace, "grafana"},
		{constants.LoggingNamespace, "loki"},
	}

	return waitForDeployments(ctx, client, components, timeout)
}

func waitForServiceMeshComponents(ctx context.Context, client *kubernetes.Client, timeout time.Duration) error {
	components := []struct {
		namespace string
		name      string
	}{
		{constants.IstioNamespace, "istiod"},
		{constants.IstioNamespace, "istio-ingressgateway"},
	}

	return waitForDeployments(ctx, client, components, timeout)
}

func waitForMonitoringComponents(ctx context.Context, client *kubernetes.Client, timeout time.Duration) error {
	components := []struct {
		namespace string
		name      string
	}{
		{constants.MonitoringNamespace, "prometheus"},
		{constants.MonitoringNamespace, "grafana"},
		{constants.MonitoringNamespace, "alertmanager"},
		{constants.LoggingNamespace, "loki"},
		{constants.TracingNamespace, "tempo"},
	}

	return waitForDeployments(ctx, client, components, timeout)
}

func waitForFullStackComponents(ctx context.Context, client *kubernetes.Client, timeout time.Duration) error {
	// This includes everything
	if err := waitForMinimalComponents(ctx, client, timeout); err != nil {
		return err
	}
	if err := waitForServiceMeshComponents(ctx, client, timeout); err != nil {
		return err
	}
	return waitForMonitoringComponents(ctx, client, timeout)
}

func waitForDeployments(ctx context.Context, client *kubernetes.Client, components []struct{ namespace, name string }, timeout time.Duration) error {
	logger := utils.NewLogger("wait")

	for _, comp := range components {
		logger.Info("Waiting for %s/%s...", comp.namespace, comp.name)
		if err := client.WaitForDeployment(ctx, comp.namespace, comp.name, timeout); err != nil {
			// Log warning but continue
			logger.Warn("Component %s/%s not ready: %v", comp.namespace, comp.name, err)
		}
	}
	return nil
}

func printScenarioAccess(scenario string) {
	logger := utils.NewLogger("access")

	logger.Info("Access services:")

	// Always available
	logger.Info("  PostgreSQL: kubectl port-forward -n %s svc/postgres 5432:5432", constants.InfraNamespace)
	logger.Info("  Redis: kubectl port-forward -n %s svc/redis 6379:6379", constants.InfraNamespace)

	switch scenario {
	case "development", "monitoring-only", "full-stack":
		logger.Info("  Grafana: kubectl port-forward -n %s svc/grafana 3000:3000 (admin/admin)", constants.MonitoringNamespace)
		logger.Info("  Prometheus: kubectl port-forward -n %s svc/prometheus 9090:9090", constants.MonitoringNamespace)
	}

	switch scenario {
	case "service-mesh", "full-stack":
		logger.Info("  Kiali: istioctl dashboard kiali")
		logger.Info("  Jaeger: kubectl port-forward -n %s svc/otel-visualizer 16686:16686", constants.TracingNamespace)
	}
}

func runConnectivityTests(ctx context.Context, client *kubernetes.Client) error {
	if client == nil {
		return fmt.Errorf("kubernetes client is nil")
	}

	logger := utils.NewLogger("test")

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
		pods, err := client.GetPods(ctx, check.namespace, check.selector)
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
}

func runMonitoringTests(ctx context.Context, client *kubernetes.Client) error {
	if client == nil {
		return fmt.Errorf("kubernetes client is nil")
	}

	logger := utils.NewLogger("test")

	// Check monitoring components
	checks := []struct {
		name      string
		namespace string
		selector  string
	}{
		{"Prometheus", constants.MonitoringNamespace, "app=prometheus"},
		{"Grafana", constants.MonitoringNamespace, "app=grafana"},
		{"Loki", constants.LoggingNamespace, "app=loki"},
		{"Alertmanager", constants.MonitoringNamespace, "app=alertmanager"},
	}

	for _, check := range checks {
		// Check deployments
		deployments, err := client.GetDeployments(ctx, check.namespace)
		if err != nil {
			logger.Warn("Failed to get deployments in %s: %v", check.namespace, err)
			continue
		}

		found := false
		for _, dep := range deployments {
			if strings.Contains(dep.Name, strings.ToLower(check.name)) {
				if dep.Status.ReadyReplicas > 0 {
					logger.Info("✓ %s: %d/%d replicas ready", check.name, dep.Status.ReadyReplicas, dep.Status.Replicas)
					found = true
				} else {
					logger.Error("✗ %s: 0/%d replicas ready", check.name, dep.Status.Replicas)
					found = true
				}
				break
			}
		}

		if !found {
			// Check statefulsets
			statefulsets, err := client.GetStatefulSets(ctx, check.namespace)
			if err == nil {
				for _, sts := range statefulsets {
					if strings.Contains(sts.Name, strings.ToLower(check.name)) {
						if sts.Status.ReadyReplicas > 0 {
							logger.Info("✓ %s: %d/%d replicas ready", check.name, sts.Status.ReadyReplicas, sts.Status.Replicas)
						} else {
							logger.Error("✗ %s: 0/%d replicas ready", check.name, sts.Status.Replicas)
						}
						break
					}
				}
			}
		}
	}

	return nil
}

func runServiceMeshTests(ctx context.Context, client *kubernetes.Client) error {
	if client == nil {
		return fmt.Errorf("kubernetes client is nil")
	}

	logger := utils.NewLogger("test")

	// Check Istio components
	checks := []struct {
		name      string
		namespace string
		selector  string
	}{
		{"Istiod", constants.IstioNamespace, "app=istiod"},
		{"Ingress Gateway", constants.IstioNamespace, "app=istio-ingressgateway"},
		{"Egress Gateway", constants.IstioNamespace, "app=istio-egressgateway"},
	}

	for _, check := range checks {
		deployments, err := client.GetDeployments(ctx, check.namespace)
		if err != nil {
			logger.Warn("Failed to get deployments in %s: %v", check.namespace, err)
			continue
		}

		found := false
		for _, dep := range deployments {
			if strings.Contains(dep.Name, strings.ToLower(strings.ReplaceAll(check.name, " ", "-"))) {
				if dep.Status.ReadyReplicas > 0 {
					logger.Info("✓ %s: %d/%d replicas ready", check.name, dep.Status.ReadyReplicas, dep.Status.Replicas)
				} else {
					logger.Error("✗ %s: 0/%d replicas ready", check.name, dep.Status.Replicas)
				}
				found = true
				break
			}
		}

		if !found {
			logger.Warn("⚠ %s: not found", check.name)
		}
	}

	// Check for Istio injection
	namespaces := []string{constants.AppNamespace, constants.InfraNamespace}
	for _, ns := range namespaces {
		exists, err := client.NamespaceExists(ctx, ns)
		if err == nil && exists {
			// This is a simplified check - in reality we'd check namespace labels
			logger.Info("✓ Namespace %s exists (check istio-injection label manually)", ns)
		}
	}

	return nil
}
