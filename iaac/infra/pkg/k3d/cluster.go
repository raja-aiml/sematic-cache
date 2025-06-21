package k3d

import (
	"context"
	"fmt"
	"time"

	k3dCluster "github.com/k3d-io/k3d/v5/pkg/client"
	"github.com/k3d-io/k3d/v5/pkg/config"
	conf "github.com/k3d-io/k3d/v5/pkg/config/v1alpha5"
	"github.com/k3d-io/k3d/v5/pkg/runtimes"
	k3d "github.com/k3d-io/k3d/v5/pkg/types"
	"k8s.io/client-go/tools/clientcmd"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
)

// ClusterOperations defines the interface for cluster management operations
// This follows the Interface Segregation Principle from CLAUDE.md
type ClusterOperations interface {
	CreateCluster(ctx context.Context) error
	DeleteCluster(ctx context.Context) error
	GetCluster(ctx context.Context) (*k3d.Cluster, error)
	IsRunning(ctx context.Context) bool
	GetKubeconfig(ctx context.Context) ([]byte, error)
}

// ClusterManager implements ClusterOperations using k3d SDK
// This follows the SDK-first approach mandated in CLAUDE.md
type ClusterManager struct {
	runtime     runtimes.Runtime
	logger      *utils.Logger
	clusterName string
	config      *conf.SimpleConfig
}

// Compile-time interface compliance check
var _ ClusterOperations = (*ClusterManager)(nil)

// NewClusterManager creates a new cluster manager using k3d SDK
func NewClusterManager(clusterName string) (*ClusterManager, error) {
	// Initialize Docker runtime (k3d uses Docker by default)
	runtime, err := runtimes.GetRuntime("docker")
	if err != nil {
		return nil, fmt.Errorf("failed to initialize Docker runtime: %w", err)
	}

	// Create simple cluster configuration using v1alpha5 config
	clusterConfig := &conf.SimpleConfig{
		Servers: 1,
		Agents:  0,
		Image:   k3d.DefaultK3sImageRepo,
		Ports: []conf.PortWithNodeFilters{
			{
				Port:        "8080:80",
				NodeFilters: []string{"loadbalancer"},
			},
			{
				Port:        "8443:443",
				NodeFilters: []string{"loadbalancer"},
			},
		},
		Options: conf.SimpleConfigOptions{
			K3dOptions: conf.SimpleConfigOptionsK3d{
				Wait:                true,
				Timeout:             5 * time.Minute,
				DisableLoadbalancer: false,
				DisableImageVolume:  false,
			},
			K3sOptions: conf.SimpleConfigOptionsK3s{
				ExtraArgs: []conf.K3sArgWithNodeFilters{
					{
						Arg:         "--disable=traefik",
						NodeFilters: []string{"server:*"},
					},
				},
			},
			KubeconfigOptions: conf.SimpleConfigOptionsKubeconfig{
				UpdateDefaultKubeconfig: true,
				SwitchCurrentContext:    true,
			},
		},
	}

	return &ClusterManager{
		runtime:     runtime,
		logger:      utils.NewLogger("k3d-sdk"),
		clusterName: clusterName,
		config:      clusterConfig,
	}, nil
}

// CreateCluster creates a k3d cluster using the SDK
func (cm *ClusterManager) CreateCluster(ctx context.Context) error {
	cm.logger.Info("Creating k3d cluster using SDK: %s", cm.clusterName)

	// Check if cluster already exists
	if cm.IsRunning(ctx) {
		cm.logger.Warn("Cluster %s already exists", cm.clusterName)
		return nil
	}

	// Validate prerequisites using SDK
	if err := cm.validatePrerequisitesSDK(ctx); err != nil {
		return fmt.Errorf("prerequisites check failed: %w", err)
	}

	// Transform simple config to cluster config
	clusterConfig, err := config.TransformSimpleToClusterConfig(ctx, cm.runtime, *cm.config, cm.clusterName)
	if err != nil {
		return fmt.Errorf("failed to transform config: %w", err)
	}

	// Create cluster using k3d cluster client
	if err := k3dCluster.ClusterRun(ctx, cm.runtime, clusterConfig); err != nil {
		return fmt.Errorf("failed to create cluster using SDK: %w", err)
	}

	cm.logger.Info("Cluster %s created successfully using SDK", cm.clusterName)
	return nil
}

// DeleteCluster deletes a k3d cluster using the SDK
func (cm *ClusterManager) DeleteCluster(ctx context.Context) error {
	cm.logger.Info("Deleting k3d cluster using SDK: %s", cm.clusterName)

	// Get cluster to delete
	cluster, err := cm.GetCluster(ctx)
	if err != nil {
		return fmt.Errorf("failed to get cluster for deletion: %w", err)
	}

	// Delete cluster using k3d cluster client
	if err := k3dCluster.ClusterDelete(ctx, cm.runtime, cluster, k3d.ClusterDeleteOpts{}); err != nil {
		return fmt.Errorf("failed to delete cluster using SDK: %w", err)
	}

	cm.logger.Info("Cluster %s deleted successfully using SDK", cm.clusterName)
	return nil
}

// GetCluster retrieves cluster information using SDK
func (cm *ClusterManager) GetCluster(ctx context.Context) (*k3d.Cluster, error) {
	// k3d may use different naming conventions:
	// 1. Original name as provided
	// 2. With "k3d-" prefix
	// 3. Default name "k3s-default" when config transformation doesn't preserve name
	clusterNames := []string{cm.clusterName, "k3d-" + cm.clusterName, "k3s-default"}

	for _, name := range clusterNames {
		cluster, err := k3dCluster.ClusterGet(ctx, cm.runtime, &k3d.Cluster{Name: name})
		if err == nil {
			return cluster, nil
		}
	}

	return nil, fmt.Errorf("failed to get cluster using SDK: cluster not found with names %v", clusterNames)
}

// GetKubeconfig retrieves kubeconfig using SDK
func (cm *ClusterManager) GetKubeconfig(ctx context.Context) ([]byte, error) {
	cluster, err := cm.GetCluster(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get cluster for kubeconfig: %w", err)
	}

	kubeconfig, err := k3dCluster.KubeconfigGet(ctx, cm.runtime, cluster)
	if err != nil {
		return nil, fmt.Errorf("failed to get kubeconfig using SDK: %w", err)
	}

	// Convert kubeconfig to bytes using clientcmd
	kubeconfigBytes, err := clientcmd.Write(*kubeconfig)
	if err != nil {
		return nil, fmt.Errorf("failed to write kubeconfig: %w", err)
	}

	return kubeconfigBytes, nil
}

// IsRunning checks if the cluster is running using SDK
func (cm *ClusterManager) IsRunning(ctx context.Context) bool {
	cluster, err := cm.GetCluster(ctx)
	if err != nil {
		return false
	}

	// Check if all nodes are running
	for _, node := range cluster.Nodes {
		if node.State.Running != true {
			return false
		}
	}

	return len(cluster.Nodes) > 0
}

// validatePrerequisitesSDK validates prerequisites using SDK methods
func (cm *ClusterManager) validatePrerequisitesSDK(ctx context.Context) error {
	// Check runtime availability (Docker)
	_, err := cm.runtime.Info()
	if err != nil {
		return fmt.Errorf("container runtime (Docker) is not available: %w", err)
	}

	cm.logger.Info("Prerequisites validated successfully using SDK")
	return nil
}
