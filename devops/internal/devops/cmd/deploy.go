package cmd

import (
	"github.com/spf13/cobra"

	"github.com/raja-aiml/sematic-cache/devops/pkg/taskcommand"
)

var (
	deployDir     string
	deployVars    map[string]string
	deployTaskCmd taskcommand.TaskCommand
)

// deployCmd represents the deploy command
var deployCmd = &cobra.Command{
	Use:   "deploy",
	Short: "Deployment tasks for the project",
	Long: `Execute deployment-related tasks using the Task SDK.

Supports sub-commands:
- k8s: Kubernetes deployment tasks
- k3d: k3d cluster management tasks  
- helm: Helm chart deployment tasks

Examples:
  devops deploy k8s deploy
  devops deploy k3d create --vars CLUSTER_NAME=dev
  devops deploy helm install --vars CHART_NAME=myapp`,
}

// deployK8sCmd represents the deploy k8s command
var deployK8sCmd = &cobra.Command{
	Use:   "k8s [task]",
	Short: "Execute Kubernetes deployment tasks",
	Long: `Execute Kubernetes deployment tasks from devops/tasks/deploy/k8s.yaml

Available tasks will be loaded from the Taskfile.

Examples:
  devops deploy k8s             # List available tasks
  devops deploy k8s deploy      # Deploy to Kubernetes
  devops deploy k8s status      # Check deployment status
  devops deploy k8s logs        # View application logs`,
	RunE: runDeployK8s,
}

// deployK3dCmd represents the deploy k3d command
var deployK3dCmd = &cobra.Command{
	Use:   "k3d [task]",
	Short: "Execute k3d cluster management tasks",
	Long: `Execute k3d cluster management tasks from devops/tasks/deploy/k3d.yaml

Available tasks will be loaded from the Taskfile.

Examples:
  devops deploy k3d             # List available tasks
  devops deploy k3d create      # Create k3d cluster
  devops deploy k3d delete      # Delete k3d cluster
  devops deploy k3d info        # Show cluster info`,
	RunE: runDeployK3d,
}

// deployHelmCmd represents the deploy helm command
var deployHelmCmd = &cobra.Command{
	Use:   "helm [task]",
	Short: "Execute Helm deployment tasks",
	Long: `Execute Helm deployment tasks from devops/tasks/deploy/helm.yaml

Available tasks will be loaded from the Taskfile.

Examples:
  devops deploy helm            # List available tasks
  devops deploy helm install   # Install Helm chart
  devops deploy helm upgrade   # Upgrade Helm release
  devops deploy helm uninstall # Uninstall Helm release`,
	RunE: runDeployHelm,
}

func runDeployK8s(cmd *cobra.Command, args []string) error {
	return deployTaskCmd.RunTask("deploy/k8s.yaml", args, deployVars, deployDir)
}

func runDeployK3d(cmd *cobra.Command, args []string) error {
	return deployTaskCmd.RunTask("deploy/k3d.yaml", args, deployVars, deployDir)
}

func runDeployHelm(cmd *cobra.Command, args []string) error {
	return deployTaskCmd.RunTask("deploy/helm.yaml", args, deployVars, deployDir)
}

func init() {
	// Initialize task command interface
	deployTaskCmd = taskcommand.NewBaseTaskCommand()

	// Add deploy command to root
	rootCmd.AddCommand(deployCmd)

	// Add subcommands to deploy
	deployCmd.AddCommand(deployK8sCmd)
	deployCmd.AddCommand(deployK3dCmd)
	deployCmd.AddCommand(deployHelmCmd)

	// Add flags
	deployCmd.PersistentFlags().StringVar(&deployDir, "dir", "", "Project directory (default: current directory)")
	deployCmd.PersistentFlags().StringToStringVar(&deployVars, "vars", map[string]string{}, "Task variables (key=value)")
}
