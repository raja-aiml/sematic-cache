package k3d

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
)

type ClusterManager struct {
	logger      *utils.Logger
	clusterName string
}

func NewClusterManager(clusterName string) *ClusterManager {
	return &ClusterManager{
		logger:      utils.NewLogger("k3d"),
		clusterName: clusterName,
	}
}

func (cm *ClusterManager) CreateCluster(ctx context.Context) error {
	cm.logger.Info("Creating k3d cluster: %s", cm.clusterName)

	// Check if cluster already exists
	if cm.IsRunning(ctx) {
		cm.logger.Warn("Cluster %s already exists", cm.clusterName)
		return nil
	}

	// Validate prerequisites
	if err := cm.validatePrerequisites(); err != nil {
		return fmt.Errorf("prerequisites check failed: %w", err)
	}

	// Create cluster using k3d CLI
	args := []string{
		"cluster", "create", cm.clusterName,
		"--servers", "1",
		"--agents", "0",
		"--api-port", "6550",
		"--port", "8080:80@loadbalancer",
		"--port", "8443:443@loadbalancer",
		"--k3s-arg", "--disable=traefik@server:*",
		"--wait",
		"--timeout", "300s",
		"--kubeconfig-update-default",
		"--kubeconfig-switch-context",
	}

	if _, err := utils.RunCommand(ctx, "k3d", args, nil); err != nil {
		return fmt.Errorf("failed to create cluster: %w", err)
	}

	cm.logger.Info("Cluster %s created successfully", cm.clusterName)
	return nil
}

func (cm *ClusterManager) DeleteCluster(ctx context.Context) error {
	cm.logger.Info("Deleting k3d cluster: %s", cm.clusterName)

	args := []string{
		"cluster", "delete", cm.clusterName,
	}

	if _, err := utils.RunCommand(ctx, "k3d", args, nil); err != nil {
		return fmt.Errorf("failed to delete cluster: %w", err)
	}

	cm.logger.Info("Cluster %s deleted successfully", cm.clusterName)
	return nil
}

func (cm *ClusterManager) GetKubeconfig(ctx context.Context) ([]byte, error) {
	args := []string{
		"kubeconfig", "get", cm.clusterName,
	}

	output, err := utils.RunCommand(ctx, "k3d", args, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get kubeconfig: %w", err)
	}

	return []byte(output), nil
}

func (cm *ClusterManager) IsRunning(ctx context.Context) bool {
	// Get cluster list in JSON format
	args := []string{
		"cluster", "list", "-o", "json",
	}

	output, err := utils.RunCommand(ctx, "k3d", args, &utils.ExecOptions{Silent: true})
	if err != nil {
		return false
	}

	// Parse JSON output
	var clusters []map[string]interface{}
	if err := json.Unmarshal([]byte(output), &clusters); err != nil {
		// Fallback to simple text parsing
		return cm.isRunningFallback(ctx)
	}

	// Check if our cluster exists and is running
	for _, cluster := range clusters {
		if name, ok := cluster["name"].(string); ok && name == cm.clusterName {
			if nodes, ok := cluster["nodes"].([]interface{}); ok {
				// Check if all nodes are running
				allRunning := true
				for _, node := range nodes {
					if nodeMap, ok := node.(map[string]interface{}); ok {
						if state, ok := nodeMap["state"].(string); ok && state != "running" {
							allRunning = false
							break
						}
					}
				}
				return allRunning
			}
		}
	}

	return false
}

func (cm *ClusterManager) isRunningFallback(ctx context.Context) bool {
	// Fallback method using text output
	args := []string{
		"cluster", "list",
	}

	output, err := utils.RunCommand(ctx, "k3d", args, &utils.ExecOptions{Silent: true})
	if err != nil {
		return false
	}

	// Check if cluster exists in the output
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		if strings.Contains(line, cm.clusterName) {
			// Simple check - if it appears in the list, assume it's running
			return true
		}
	}

	return false
}

func (cm *ClusterManager) validatePrerequisites() error {
	// Check if Docker is running
	if !utils.CommandExists("docker") {
		return fmt.Errorf("docker is not installed")
	}

	// Check if docker daemon is running
	if _, err := utils.RunCommand(context.Background(), "docker", []string{"info"}, &utils.ExecOptions{Silent: true}); err != nil {
		return fmt.Errorf("docker daemon is not running")
	}

	// Check if k3d is installed
	if !utils.CommandExists("k3d") {
		return fmt.Errorf("k3d is not installed")
	}

	return nil
}
