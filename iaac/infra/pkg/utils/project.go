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
// If overridePath is provided and exists, it returns that path directly
func GetKustomizePath(overlay string, overridePath ...string) (string, error) {
	// Check if override path is provided
	if len(overridePath) > 0 && overridePath[0] != "" {
		// Check if the override path exists
		if _, err := os.Stat(overridePath[0]); err == nil {
			return overridePath[0], nil
		}
		// If provided but doesn't exist, return error
		return "", fmt.Errorf("kustomize path %s does not exist", overridePath[0])
	}

	// Get blueprint path from environment or use default
	blueprintPath := os.Getenv("IAAC_BLUEPRINT_PATH")
	if blueprintPath == "" {
		blueprintPath = "iaac/blueprint"
	}

	// If blueprint path is absolute, use it as is
	if filepath.IsAbs(blueprintPath) {
		return getKustomizePathFromBase(blueprintPath, overlay)
	}

	// Otherwise, find the main project root (not the current module root)
	root, err := findMainProjectRoot()
	if err != nil {
		return "", err
	}

	return getKustomizePathFromBase(filepath.Join(root, blueprintPath), overlay)
}

// findMainProjectRoot finds the main project root by looking for the blueprint directory
func findMainProjectRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get working directory: %w", err)
	}

	// Navigate up to find the blueprint directory
	for i := 0; i < maxDepth; i++ {
		// Check if blueprint directory exists
		blueprintPath := filepath.Join(dir, "iaac", "blueprint")
		if _, err := os.Stat(blueprintPath); err == nil {
			return dir, nil
		}

		// Also check if we're in iaac/infra directory
		if filepath.Base(dir) == "infra" && filepath.Base(filepath.Dir(dir)) == "iaac" {
			mainRoot := filepath.Dir(filepath.Dir(dir))
			blueprintPath = filepath.Join(mainRoot, "iaac", "blueprint")
			if _, err := os.Stat(blueprintPath); err == nil {
				return mainRoot, nil
			}
		}

		// Also check if we're already in iaac directory
		if filepath.Base(dir) == "iaac" {
			parent := filepath.Dir(dir)
			blueprintPath = filepath.Join(parent, "iaac", "blueprint")
			if _, err := os.Stat(blueprintPath); err == nil {
				return parent, nil
			}
		}

		// Move up one directory
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	return "", fmt.Errorf("could not find project root with blueprint directory")
}

// getKustomizePathFromBase constructs the kustomize path from a base blueprint path
func getKustomizePathFromBase(blueprintBase string, overlay string) (string, error) {
	// Determine if this is for app or infra based on the overlay path
	var basePath string
	if filepath.Base(overlay) == "app" || filepath.Dir(overlay) == "local" && filepath.Base(overlay) == "app" {
		// This is for app deployment (e.g., "local/app")
		overlayDir := filepath.Dir(overlay) // Get "local" from "local/app"
		if overlayDir == "." {
			overlayDir = "local"
		}
		basePath = filepath.Join(blueprintBase, "app", "overlays", overlayDir)
	} else {
		// This is for infra deployment
		basePath = filepath.Join(blueprintBase, "infra", "overlays", overlay)
	}

	// Check if overlay exists
	if _, err := os.Stat(basePath); os.IsNotExist(err) {
		// For app, use app base; for infra, use infra base
		if filepath.Base(overlay) == "app" || filepath.Dir(overlay) == "local" && filepath.Base(overlay) == "app" {
			return filepath.Join(blueprintBase, "app", "base"), nil
		}
		return filepath.Join(blueprintBase, "infra", "base"), nil
	}

	return basePath, nil
}
