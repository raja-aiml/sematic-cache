package utils

import (
	"fmt"
	"os"
	"path/filepath"
)

const maxDepth = 10 // Maximum depth to search for go.mod

// FindProjectRoot finds the project root by looking for go.mod file
func FindProjectRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get working directory: %w", err)
	}

	depth := 0
	for {
		// Check if go.mod exists in this directory
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}

		// Check depth limit
		depth++
		if depth > maxDepth {
			return "", fmt.Errorf("could not find project root (no go.mod found within %d levels)", maxDepth)
		}

		// Move up one directory
		parent := filepath.Dir(dir)
		if parent == dir {
			// Reached root directory
			return "", fmt.Errorf("could not find project root (no go.mod found)")
		}
		dir = parent
	}
}

// GetDeployPath returns the path to the deploy directory
func GetDeployPath() (string, error) {
	root, err := FindProjectRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "deploy"), nil
}

// GetKustomizePath returns the path to the kustomize overlays
func GetKustomizePath(overlay string) (string, error) {
	root, err := FindProjectRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "deploy", "k8s", "overlays", overlay), nil
}
