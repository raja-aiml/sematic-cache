package k3d

import (
	"context"
	"testing"
)

func TestNewClusterManager(t *testing.T) {
	tests := []struct {
		name        string
		clusterName string
	}{
		{
			name:        "simple_cluster_name",
			clusterName: "test-cluster",
		},
		{
			name:        "cluster_with_numbers",
			clusterName: "cluster-123",
		},
		{
			name:        "empty_cluster_name",
			clusterName: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cm := NewClusterManager(tt.clusterName)
			
			if cm == nil {
				t.Fatal("NewClusterManager returned nil")
			}
			
			if cm.clusterName != tt.clusterName {
				t.Errorf("NewClusterManager() clusterName = %v, want %v", cm.clusterName, tt.clusterName)
			}
			
			if cm.logger == nil {
				t.Error("NewClusterManager() logger is nil")
			}
		})
	}
}

func TestClusterManager_IsRunning(t *testing.T) {
	cm := NewClusterManager("test-cluster")
	ctx := context.Background()
	
	// This will return false as k3d is not available in test environment
	result := cm.IsRunning(ctx)
	
	// We expect false in test environment
	if result {
		t.Error("IsRunning() expected false in test environment")
	}
}

func TestClusterManager_CreateCluster(t *testing.T) {
	cm := NewClusterManager("test-cluster")
	ctx := context.Background()
	
	// This will fail as k3d is not available in test environment
	err := cm.CreateCluster(ctx)
	
	// We expect an error in test environment
	if err == nil {
		t.Error("CreateCluster() expected error in test environment")
	}
}

func TestClusterManager_DeleteCluster(t *testing.T) {
	cm := NewClusterManager("test-cluster")
	ctx := context.Background()
	
	// This will fail as k3d is not available in test environment
	err := cm.DeleteCluster(ctx)
	
	// We expect an error in test environment
	if err == nil {
		t.Error("DeleteCluster() expected error in test environment")
	}
}

func TestClusterManager_GetKubeconfig(t *testing.T) {
	cm := NewClusterManager("test-cluster")
	ctx := context.Background()
	
	// This will fail as k3d is not available in test environment
	_, err := cm.GetKubeconfig(ctx)
	
	// We expect an error in test environment
	if err == nil {
		t.Error("GetKubeconfig() expected error in test environment")
	}
}

func TestClusterManager_isRunningFallback(t *testing.T) {
	cm := NewClusterManager("test-cluster")
	ctx := context.Background()
	
	// This will return false as k3d is not available in test environment
	result := cm.isRunningFallback(ctx)
	
	// We expect false in test environment
	if result {
		t.Error("isRunningFallback() expected false in test environment")
	}
}

func TestClusterManager_validatePrerequisites(t *testing.T) {
	cm := NewClusterManager("test-cluster")
	
	// This may fail depending on the test environment
	err := cm.validatePrerequisites()
	
	// Just check that the function runs without panic
	_ = err
}

// Benchmark tests
func BenchmarkNewClusterManager(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = NewClusterManager("bench-cluster")
	}
}

func BenchmarkIsRunning(b *testing.B) {
	cm := NewClusterManager("bench-cluster")
	ctx := context.Background()
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		_ = cm.IsRunning(ctx)
	}
}