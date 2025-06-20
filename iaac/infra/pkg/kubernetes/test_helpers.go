package kubernetes

import (
	"os"
	"testing"
)

// SetupTestEnvironment sets up the environment for Kubernetes client tests
// to suppress warnings about missing kubeconfig or in-cluster config
func SetupTestEnvironment(t *testing.T) func() {
	t.Helper()

	// Save original values
	origServiceHost := os.Getenv("KUBERNETES_SERVICE_HOST")
	origServicePort := os.Getenv("KUBERNETES_SERVICE_PORT")

	// Clear these to prevent in-cluster config attempts
	os.Setenv("KUBERNETES_SERVICE_HOST", "")
	os.Setenv("KUBERNETES_SERVICE_PORT", "")

	// Return cleanup function
	return func() {
		os.Setenv("KUBERNETES_SERVICE_HOST", origServiceHost)
		os.Setenv("KUBERNETES_SERVICE_PORT", origServicePort)
	}
}

// NewTestClient creates a client for testing that expects to fail
// This is useful for tests that don't need a real Kubernetes connection
func NewTestClient(t *testing.T) (*Client, error) {
	t.Helper()
	cleanup := SetupTestEnvironment(t)
	defer cleanup()

	// Try to create a client with a non-existent kubeconfig
	return NewClient("/tmp/non-existent-kubeconfig-for-testing")
}
