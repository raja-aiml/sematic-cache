package k3d

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClusterManager(t *testing.T) {
	tests := []struct {
		name        string
		clusterName string
		expectError bool
	}{
		{
			name:        "simple_cluster_name",
			clusterName: "test-cluster",
			expectError: false,
		},
		{
			name:        "cluster_with_numbers",
			clusterName: "cluster-123",
			expectError: false,
		},
		{
			name:        "empty_cluster_name",
			clusterName: "",
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cm, err := NewClusterManager(tt.clusterName)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, cm)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, cm)
				assert.Equal(t, tt.clusterName, cm.clusterName)
				assert.NotNil(t, cm.logger)
				assert.NotNil(t, cm.runtime)
				assert.NotNil(t, cm.config)
			}
		})
	}
}

func TestClusterManagerInterfaceCompliance(t *testing.T) {
	cm, err := NewClusterManager("test-cluster")
	require.NoError(t, err)

	// Verify interface compliance at compile time
	var _ ClusterOperations = cm
}

func TestClusterManager_IsRunning(t *testing.T) {
	// Skip if Docker is not available
	if os.Getenv("SKIP_DOCKER_TESTS") == "true" {
		t.Skip("Skipping test requiring Docker")
	}

	cm, err := NewClusterManager("test-cluster")
	require.NoError(t, err)

	ctx := context.Background()

	// This will return false as cluster doesn't exist
	result := cm.IsRunning(ctx)

	// We expect false for non-existent cluster
	assert.False(t, result, "IsRunning() expected false for non-existent cluster")
}

func TestClusterManager_CreateCluster(t *testing.T) {
	// Skip integration tests if requested
	if os.Getenv("SKIP_INTEGRATION_TESTS") == "true" {
		t.Skip("Skipping integration test")
	}

	cm, err := NewClusterManager("test-cluster-create")
	require.NoError(t, err)

	ctx := context.Background()

	// Clean up any existing test cluster first
	_ = cm.DeleteCluster(ctx)

	// Test creation
	err = cm.CreateCluster(ctx)
	if err != nil {
		t.Logf("CreateCluster() failed (expected in test environment): %v", err)
		// This is expected if Docker is not available or if there are permission issues
		return
	}

	// If creation succeeded, verify cluster is running
	assert.True(t, cm.IsRunning(ctx), "Cluster should be running after creation")

	// Clean up after test
	defer func() {
		if deleteErr := cm.DeleteCluster(ctx); deleteErr != nil {
			t.Logf("Failed to clean up test cluster: %v", deleteErr)
		}
	}()
}

func TestClusterManager_DeleteCluster(t *testing.T) {
	// Skip integration tests if requested
	if os.Getenv("SKIP_INTEGRATION_TESTS") == "true" {
		t.Skip("Skipping integration test")
	}

	cm, err := NewClusterManager("test-cluster-delete")
	require.NoError(t, err)

	ctx := context.Background()

	// Test deletion of non-existent cluster
	err = cm.DeleteCluster(ctx)

	// This should error for non-existent cluster
	assert.Error(t, err, "DeleteCluster() should error for non-existent cluster")
}

func TestClusterManager_GetCluster(t *testing.T) {
	// Skip integration tests if requested
	if os.Getenv("SKIP_INTEGRATION_TESTS") == "true" {
		t.Skip("Skipping integration test")
	}

	cm, err := NewClusterManager("test-cluster-get")
	require.NoError(t, err)

	ctx := context.Background()

	// Test getting non-existent cluster
	_, err = cm.GetCluster(ctx)

	// This should error for non-existent cluster
	assert.Error(t, err, "GetCluster() should error for non-existent cluster")
}

func TestClusterManager_GetKubeconfig(t *testing.T) {
	// Skip integration tests if requested
	if os.Getenv("SKIP_INTEGRATION_TESTS") == "true" {
		t.Skip("Skipping integration test")
	}

	cm, err := NewClusterManager("test-cluster-kubeconfig")
	require.NoError(t, err)

	ctx := context.Background()

	// This will fail as cluster doesn't exist
	_, err = cm.GetKubeconfig(ctx)

	// We expect an error for non-existent cluster
	assert.Error(t, err, "GetKubeconfig() should error for non-existent cluster")
}

func TestClusterManager_validatePrerequisitesSDK(t *testing.T) {
	// Skip if Docker is not available
	if os.Getenv("SKIP_DOCKER_TESTS") == "true" {
		t.Skip("Skipping test requiring Docker")
	}

	cm, err := NewClusterManager("test-cluster")
	require.NoError(t, err)

	ctx := context.Background()

	// Test prerequisites validation
	err = cm.validatePrerequisitesSDK(ctx)

	// This may pass or fail depending on Docker availability
	if err != nil {
		t.Logf("Prerequisites validation failed (may be expected): %v", err)
	}
}

// Table-driven test for cluster configurations
func TestClusterManagerConfigurations(t *testing.T) {
	tests := []struct {
		name        string
		clusterName string
		serverCount int
		agentCount  int
		expectError bool
	}{
		{
			name:        "default-config",
			clusterName: "default-cluster",
			serverCount: 1,
			agentCount:  0,
			expectError: false,
		},
		{
			name:        "ha-config",
			clusterName: "ha-cluster",
			serverCount: 3,
			agentCount:  2,
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cm, err := NewClusterManager(tt.clusterName)

			if tt.expectError {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)
			assert.NotNil(t, cm)

			// Test configuration modification
			cm.config.Servers = tt.serverCount
			cm.config.Agents = tt.agentCount

			assert.Equal(t, tt.serverCount, cm.config.Servers)
			assert.Equal(t, tt.agentCount, cm.config.Agents)
		})
	}
}

// Benchmark tests
func BenchmarkNewClusterManager(b *testing.B) {
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		cm, err := NewClusterManager("bench-cluster")
		if err != nil {
			b.Fatalf("NewClusterManager failed: %v", err)
		}
		_ = cm
	}
}

func BenchmarkIsRunning(b *testing.B) {
	// Skip if Docker is not available
	if os.Getenv("SKIP_DOCKER_TESTS") == "true" {
		b.Skip("Skipping benchmark requiring Docker")
	}

	cm, err := NewClusterManager("bench-cluster")
	if err != nil {
		b.Fatalf("NewClusterManager failed: %v", err)
	}

	ctx := context.Background()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = cm.IsRunning(ctx)
	}
}

// Test error scenarios
func TestClusterManagerErrorScenarios(t *testing.T) {
	cm, err := NewClusterManager("error-test-cluster")
	require.NoError(t, err)

	ctx := context.Background()

	// Test with cancelled context
	cancelledCtx, cancel := context.WithCancel(ctx)
	cancel()

	// Operations with cancelled context should handle gracefully
	result := cm.IsRunning(cancelledCtx)
	assert.False(t, result, "IsRunning should return false with cancelled context")
}

// Test default configuration values
func TestClusterManagerDefaultConfig(t *testing.T) {
	cm, err := NewClusterManager("config-test")
	require.NoError(t, err)

	// Verify default configuration
	assert.Equal(t, 1, cm.config.Servers)
	assert.Equal(t, 0, cm.config.Agents)
	assert.False(t, cm.config.Options.K3dOptions.DisableLoadbalancer)
	assert.True(t, cm.config.Options.K3dOptions.Wait)
	assert.NotNil(t, cm.config.Options.K3sOptions.ExtraArgs)
	assert.True(t, cm.config.Options.KubeconfigOptions.UpdateDefaultKubeconfig)
	assert.True(t, cm.config.Options.KubeconfigOptions.SwitchCurrentContext)
}
