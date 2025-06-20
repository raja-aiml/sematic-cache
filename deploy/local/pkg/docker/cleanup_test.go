// Package docker test cleanup utilities
// This file ensures that all test Docker images and containers are cleaned up
// before and after running tests to prevent resource leaks.
package docker

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"testing"
)

// TestMain runs cleanup before and after all tests
func TestMain(m *testing.M) {
	// Run cleanup before tests (without testing.T)
	cleanupBeforeTests()

	// Run tests
	code := m.Run()

	// Run cleanup after tests
	cleanupAfterTests()

	// Exit with test result code
	os.Exit(code)
}

// cleanupBeforeTests removes test resources before running tests
func cleanupBeforeTests() {
	// Check if Docker is available
	if err := exec.Command("docker", "version").Run(); err != nil {
		return
	}

	ctx := context.Background()

	// Remove test images
	testImagePatterns := []string{"test:", "test-", "bench:", "nonexistent:"}
	cmd := exec.CommandContext(ctx, "docker", "images", "--format", "{{.Repository}}:{{.Tag}}")
	if output, err := cmd.Output(); err == nil {
		images := strings.Split(string(output), "\n")
		for _, img := range images {
			img = strings.TrimSpace(img)
			if img == "" {
				continue
			}
			for _, pattern := range testImagePatterns {
				if strings.HasPrefix(img, pattern) {
					if err := exec.CommandContext(ctx, "docker", "rmi", "-f", img).Run(); err != nil {
						// Ignore error - cleanup effort
						_ = err
					}
					break
				}
			}
		}
	}

	// Remove test containers
	testContainerPatterns := []string{"test-container", "container-123"}
	cmd = exec.CommandContext(ctx, "docker", "ps", "-a", "--format", "{{.Names}}")
	if output, err := cmd.Output(); err == nil {
		containers := strings.Split(string(output), "\n")
		for _, container := range containers {
			container = strings.TrimSpace(container)
			if container == "" {
				continue
			}
			for _, pattern := range testContainerPatterns {
				if strings.Contains(container, pattern) {
					if err := exec.CommandContext(ctx, "docker", "rm", "-f", container).Run(); err != nil {
						// Ignore error - cleanup effort
						_ = err
					}
					break
				}
			}
		}
	}
}

// cleanupAfterTests performs the same cleanup after tests
func cleanupAfterTests() {
	cleanupBeforeTests()
}
