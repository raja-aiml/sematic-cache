package blueprint

import (
	"fmt"
	"path/filepath"
	"sync"
)

// Adapter provides backward compatibility with the old constants-based approach
type Adapter struct {
	manager *Manager
	mu      sync.RWMutex
}

// globalAdapter is the singleton adapter instance
var (
	globalAdapter *Adapter
	adapterOnce   sync.Once
)

// GetAdapter returns the global adapter instance
func GetAdapter() *Adapter {
	adapterOnce.Do(func() {
		manager, _ := GetGlobalManager()
		globalAdapter = &Adapter{
			manager: manager,
		}
	})
	return globalAdapter
}

// GetScenarioPath returns the path to a scenario
func (a *Adapter) GetScenarioPath(scenario string) string {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if a.manager != nil {
		if path, err := a.manager.GetScenarioPath(scenario); err == nil {
			return path
		}
	}

	// Fallback to hardcoded path
	return filepath.Join("iaac", "blueprint", "scenarios", scenario)
}

// GetModulePath returns the path to a module
func (a *Adapter) GetModulePath(module string) string {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if a.manager != nil {
		if path, err := a.manager.GetModulePath(module); err == nil {
			return path
		}
	}

	// Fallback to hardcoded path
	return filepath.Join("iaac", "blueprint", "infra", "modules", module)
}

// GetOverlayPath returns the path to an overlay
func (a *Adapter) GetOverlayPath(overlay string) string {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if a.manager != nil {
		if path, err := a.manager.GetOverlayPath(overlay); err == nil {
			return path
		}
	}

	// Fallback to hardcoded path
	return filepath.Join("iaac", "blueprint", "infra", "overlays", overlay)
}

// GetScenarioNamespaces returns expected namespaces for a scenario
func (a *Adapter) GetScenarioNamespaces(scenario string) []string {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if a.manager != nil {
		if namespaces, err := a.manager.GetScenarioNamespaces(scenario); err == nil {
			return namespaces
		}
	}

	// Fallback to hardcoded namespaces
	switch scenario {
	case "minimal":
		return []string{"infra", "app"}
	case "development":
		return []string{"infra", "app", "dev-tools"}
	case "service-mesh":
		return []string{"infra", "app", "istio-system", "istio-ingress"}
	case "monitoring-only":
		return []string{"infra", "app", "monitoring", "logging"}
	case "full-stack":
		return []string{"infra", "app", "istio-system", "istio-ingress", "monitoring", "logging", "dev-tools"}
	default:
		return []string{"infra", "app"}
	}
}

// ListScenarios returns available scenarios
func (a *Adapter) ListScenarios() []string {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if a.manager != nil {
		return a.manager.GetConfig().ListScenarios()
	}

	// Fallback to hardcoded list
	return []string{
		"minimal",
		"development",
		"service-mesh",
		"monitoring-only",
		"full-stack",
	}
}

// ValidateScenario checks if a scenario exists
func (a *Adapter) ValidateScenario(scenario string) error {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if a.manager != nil {
		return a.manager.GetConfig().ValidateScenario(scenario)
	}

	// Fallback to hardcoded validation
	validScenarios := map[string]bool{
		"minimal":         true,
		"development":     true,
		"service-mesh":    true,
		"monitoring-only": true,
		"full-stack":      true,
	}

	if !validScenarios[scenario] {
		return fmt.Errorf("unknown scenario: %s", scenario)
	}

	return nil
}

// GetBlueprintPath returns the path to a blueprint component
func (a *Adapter) GetBlueprintPath(component string) string {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if a.manager != nil {
		basePath := a.manager.GetBasePath()
		return filepath.Join(basePath, component)
	}

	// Fallback
	return filepath.Join("iaac", "blueprint", component)
}

// GetInfraPath returns the infrastructure path
func (a *Adapter) GetInfraPath() string {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if a.manager != nil {
		return a.manager.GetConfig().GetInfraPath()
	}

	// Fallback
	return filepath.Join("iaac", "blueprint", "infra")
}

// GetAppPath returns the application path
func (a *Adapter) GetAppPath() string {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if a.manager != nil {
		return a.manager.GetConfig().GetAppPath()
	}

	// Fallback
	return filepath.Join("iaac", "blueprint", "app")
}

// Package-level convenience functions that use the global adapter

// GetScenarioPath returns the path to a scenario
func GetScenarioPath(scenario string) string {
	return GetAdapter().GetScenarioPath(scenario)
}

// GetModulePath returns the path to a module
func GetModulePath(module string) string {
	return GetAdapter().GetModulePath(module)
}

// GetOverlayPath returns the path to an overlay
func GetOverlayPath(overlay string) string {
	return GetAdapter().GetOverlayPath(overlay)
}

// GetScenarioNamespaces returns expected namespaces for a scenario
func GetScenarioNamespaces(scenario string) []string {
	return GetAdapter().GetScenarioNamespaces(scenario)
}

// ListScenarios returns available scenarios
func ListScenarios() []string {
	return GetAdapter().ListScenarios()
}

// ValidateScenario checks if a scenario exists
func ValidateScenario(scenario string) error {
	return GetAdapter().ValidateScenario(scenario)
}
