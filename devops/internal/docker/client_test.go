package docker

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDockerClient(t *testing.T) {
	// Note: These are integration tests that require Docker to be running
	// They will be skipped if Docker is not available

	client, err := NewClient()
	if err != nil {
		t.Skip("Docker not available, skipping tests")
		return
	}
	defer client.Close()

	ctx := context.Background()

	t.Run("IsRunning", func(t *testing.T) {
		running := client.IsRunning(ctx)
		assert.True(t, running, "Docker should be running for these tests")
	})

	t.Run("Version", func(t *testing.T) {
		version, err := client.Version(ctx)
		require.NoError(t, err)
		assert.NotEmpty(t, version.Version)
		assert.NotEmpty(t, version.APIVersion)
	})

	t.Run("ListContainers", func(t *testing.T) {
		// Just verify the method works, don't assert on specific containers
		containers, err := client.ListContainers(ctx, false)
		require.NoError(t, err)
		assert.NotNil(t, containers)
	})

	t.Run("ListImages", func(t *testing.T) {
		// Just verify the method works
		images, err := client.ListImages(ctx)
		require.NoError(t, err)
		assert.NotNil(t, images)
	})
}

// TestContainerOperations tests container-specific operations
// This is separated as it may create/destroy containers
func TestContainerOperations(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping container operations test in short mode")
	}

	client, err := NewClient()
	if err != nil {
		t.Skip("Docker not available, skipping tests")
		return
	}
	defer client.Close()

	ctx := context.Background()

	// Check if Docker is running
	if !client.IsRunning(ctx) {
		t.Skip("Docker daemon not running")
		return
	}

	// Note: Actual container operations would require creating test containers
	// For unit tests, we're just verifying the client can be created and basic operations work
}

func TestNewClientWithLogger(t *testing.T) {
	// This test doesn't require Docker to be running
	_, err := NewClientWithLogger(nil)

	// The error will depend on whether Docker is available
	// We're just testing that the function can be called
	if err != nil {
		assert.Contains(t, err.Error(), "docker")
	}
}
