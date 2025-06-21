package docker

import (
	"context"
	"testing"
	"time"

	"github.com/docker/docker/client"
)

// isDockerAvailable checks if Docker daemon is available
func isDockerAvailable() bool {
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return false
	}
	defer cli.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_, err = cli.Ping(ctx)
	return err == nil
}

func TestNewBuilder(t *testing.T) {
	if !isDockerAvailable() {
		t.Skip("Docker daemon not available")
	}

	builder, err := NewBuilder()
	if err != nil {
		t.Fatalf("Failed to create unified builder: %v", err)
	}
	defer builder.Close()

	if builder.client == nil {
		t.Error("Expected client to be initialized")
	}

	if builder.logger == nil {
		t.Error("Expected logger to be initialized")
	}
}

func TestBuilderImageOperations(t *testing.T) {
	if !isDockerAvailable() {
		t.Skip("Docker daemon not available")
	}

	builder, err := NewBuilder()
	if err != nil {
		t.Fatalf("Failed to create unified builder: %v", err)
	}
	defer builder.Close()

	ctx := context.Background()

	// Test listing images
	images, err := builder.ListImages(ctx)
	if err != nil {
		t.Errorf("Failed to list images: %v", err)
	}

	t.Logf("Found %d images", len(images))
}
