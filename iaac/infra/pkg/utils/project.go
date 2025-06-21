package utils

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/blueprint"
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

	// Try to use blueprint adapter first
	adapter := blueprint.GetAdapter()
	if adapter != nil {
		// Check if this is an app overlay
		if filepath.Base(overlay) == "app" || (filepath.Dir(overlay) == "local" && filepath.Base(overlay) == "app") {
			appPath := adapter.GetAppPath()
			overlayDir := filepath.Dir(overlay)
			if overlayDir == "." {
				overlayDir = "local"
			}
			kustomizePath := filepath.Join(appPath, "overlays", overlayDir)
			if _, err := os.Stat(kustomizePath); err == nil {
				return kustomizePath, nil
			}
			// Fall back to app base
			return filepath.Join(appPath, "base"), nil
		}

		// For infra overlays, use the overlay path directly
		overlayPath := adapter.GetOverlayPath(overlay)
		if _, err := os.Stat(overlayPath); err == nil {
			return overlayPath, nil
		}
	}

	// Fall back to legacy path resolution for backward compatibility
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
	// Try to find blueprint config using the manager's method
	configPath, err := blueprint.FindBlueprintConfig(".")
	if err == nil {
		// Extract the project root from the config path
		// The config is typically at iaac/blueprint/.blueprint.yaml
		dir := filepath.Dir(configPath)
		for i := 0; i < maxDepth; i++ {
			// Check if we've found the project root (contains iaac directory)
			if filepath.Base(dir) == "blueprint" && filepath.Base(filepath.Dir(dir)) == "iaac" {
				return filepath.Dir(filepath.Dir(dir)), nil
			}
			// Check if config is at project root
			if _, err := os.Stat(filepath.Join(dir, "iaac", "blueprint")); err == nil {
				return dir, nil
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}

	// Fall back to legacy directory search
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
// This is kept for backward compatibility when blueprint configuration is not available
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

// GetScenarioPath returns the path to a scenario using the blueprint configuration
func GetScenarioPath(scenario string) (string, error) {
	adapter := blueprint.GetAdapter()
	if adapter != nil {
		path := adapter.GetScenarioPath(scenario)
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
		return "", fmt.Errorf("scenario path %s does not exist", path)
	}

	// Fall back to legacy path
	root, err := findMainProjectRoot()
	if err != nil {
		return "", err
	}

	path := filepath.Join(root, "iaac", "blueprint", "scenarios", scenario)
	if _, err := os.Stat(path); err != nil {
		return "", fmt.Errorf("scenario %s not found at %s", scenario, path)
	}

	return path, nil
}

// GetModulePath returns the path to a module using the blueprint configuration
func GetModulePath(module string) (string, error) {
	adapter := blueprint.GetAdapter()
	if adapter != nil {
		path := adapter.GetModulePath(module)
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
		return "", fmt.Errorf("module path %s does not exist", path)
	}

	// Fall back to legacy path
	root, err := findMainProjectRoot()
	if err != nil {
		return "", err
	}

	path := filepath.Join(root, "iaac", "blueprint", "infra", "modules", module)
	if _, err := os.Stat(path); err != nil {
		return "", fmt.Errorf("module %s not found at %s", module, path)
	}

	return path, nil
}

// GetBlueprintPath returns the path to a blueprint component
func GetBlueprintPath(component string) (string, error) {
	adapter := blueprint.GetAdapter()
	if adapter != nil {
		path := adapter.GetBlueprintPath(component)
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}

	// Fall back to legacy path resolution
	root, err := findMainProjectRoot()
	if err != nil {
		return "", err
	}

	return filepath.Join(root, "iaac", "blueprint", component), nil
}

// ListAvailableScenarios returns a list of available scenarios
func ListAvailableScenarios() []string {
	adapter := blueprint.GetAdapter()
	if adapter != nil {
		return adapter.ListScenarios()
	}

	// Fall back to empty list
	return []string{}
}

// ValidateScenario checks if a scenario is valid
func ValidateScenario(scenario string) error {
	adapter := blueprint.GetAdapter()
	if adapter != nil {
		return adapter.ValidateScenario(scenario)
	}

	// Fall back to checking if the scenario directory exists
	_, err := GetScenarioPath(scenario)
	return err
}
