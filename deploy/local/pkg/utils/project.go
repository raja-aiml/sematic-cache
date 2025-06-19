package utils

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

var (
	projectRootCache string
	projectRootOnce  sync.Once
	projectRootErr   error
)

// FindProjectRoot finds the project root by looking for go.mod file
// Results are cached for subsequent calls
func FindProjectRoot() (string, error) {
	projectRootOnce.Do(func() {
		dir, err := os.Getwd()
		if err != nil {
			projectRootErr = fmt.Errorf("failed to get working directory: %w", err)
			return
		}

		for {
			// Check if go.mod exists in this directory
			if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
				projectRootCache = dir
				return
			}

			// Move up one directory
			parent := filepath.Dir(dir)
			if parent == dir {
				// Reached root directory
				projectRootErr = fmt.Errorf("could not find project root (no go.mod found)")
				return
			}
			dir = parent
		}
	})

	return projectRootCache, projectRootErr
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
