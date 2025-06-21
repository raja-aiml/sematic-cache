package blueprint

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// Manager manages blueprint operations
type Manager struct {
	config     *Config
	configPath string
	mu         sync.RWMutex
}

// NewManager creates a new blueprint manager
func NewManager(configPath string) (*Manager, error) {
	config, err := LoadConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load blueprint config: %w", err)
	}

	return &Manager{
		config:     config,
		configPath: configPath,
	}, nil
}

// FindBlueprintConfig searches for blueprint configuration file
func FindBlueprintConfig(startPath string) (string, error) {
	// List of possible config file names
	configNames := []string{
		".blueprint.yaml",
		".blueprint.yml",
		"blueprint.yaml",
		"blueprint.yml",
		"blueprint-config.yaml",
		"blueprint-config.yml",
	}

	// Convert to absolute path
	absPath, err := filepath.Abs(startPath)
	if err != nil {
		return "", err
	}

	// Walk up the directory tree
	current := absPath
	for {
		// Check each possible config name
		for _, name := range configNames {
			configPath := filepath.Join(current, name)
			if _, err := os.Stat(configPath); err == nil {
				return configPath, nil
			}

			// Also check in iaac/blueprint subdirectory
			blueprintPath := filepath.Join(current, "iaac", "blueprint", name)
			if _, err := os.Stat(blueprintPath); err == nil {
				return blueprintPath, nil
			}

			// Check in blueprint subdirectory
			blueprintPath = filepath.Join(current, "blueprint", name)
			if _, err := os.Stat(blueprintPath); err == nil {
				return blueprintPath, nil
			}
		}

		// Move up one directory
		parent := filepath.Dir(current)
		if parent == current {
			// Reached root
			break
		}
		current = parent
	}

	return "", fmt.Errorf("blueprint configuration not found")
}

// GetConfig returns the current configuration
func (m *Manager) GetConfig() *Config {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.config
}

// ReloadConfig reloads the configuration from disk
func (m *Manager) ReloadConfig() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	config, err := LoadConfig(m.configPath)
	if err != nil {
		return err
	}

	m.config = config
	return nil
}

// ValidateStructure validates the blueprint directory structure
func (m *Manager) ValidateStructure() error {
	config := m.GetConfig()

	// Check required directories
	if config.Validation.RequiredDirectories != nil {
		for _, dir := range config.Validation.RequiredDirectories {
			path := filepath.Join(config.Paths.Base, dir)
			if _, err := os.Stat(path); os.IsNotExist(err) {
				return fmt.Errorf("required directory missing: %s", dir)
			}
		}
	}

	// Check required files
	if config.Validation.RequiredFiles != nil {
		for _, file := range config.Validation.RequiredFiles {
			path := filepath.Join(config.Paths.Base, file)
			if _, err := os.Stat(path); os.IsNotExist(err) {
				return fmt.Errorf("required file missing: %s", file)
			}
		}
	}

	return nil
}

// GetScenarioPath returns the path to a scenario
func (m *Manager) GetScenarioPath(scenario string) (string, error) {
	config := m.GetConfig()

	if err := config.ValidateScenario(scenario); err != nil {
		return "", err
	}

	return config.GetScenarioPath(scenario), nil
}

// GetModulePath returns the path to a module
func (m *Manager) GetModulePath(module string) (string, error) {
	config := m.GetConfig()

	if _, ok := config.Modules[module]; !ok {
		return "", fmt.Errorf("unknown module '%s', available: %v", module, config.ListModules())
	}

	return config.GetModulePath(module), nil
}

// GetOverlayPath returns the path to an overlay
func (m *Manager) GetOverlayPath(overlay string) (string, error) {
	config := m.GetConfig()

	if _, ok := config.Overlays[overlay]; !ok {
		return "", fmt.Errorf("unknown overlay '%s', available: %v", overlay, config.ListOverlays())
	}

	return config.GetOverlayPath(overlay), nil
}

// GetScenarioComponents returns the components for a scenario
func (m *Manager) GetScenarioComponents(scenario string) ([]string, error) {
	config := m.GetConfig()

	sc, ok := config.Scenarios[scenario]
	if !ok {
		return nil, fmt.Errorf("unknown scenario '%s'", scenario)
	}

	return sc.Components, nil
}

// GetScenarioNamespaces returns the namespaces for a scenario
func (m *Manager) GetScenarioNamespaces(scenario string) ([]string, error) {
	config := m.GetConfig()

	sc, ok := config.Scenarios[scenario]
	if !ok {
		return nil, fmt.Errorf("unknown scenario '%s'", scenario)
	}

	if len(sc.Namespaces) > 0 {
		return sc.Namespaces, nil
	}

	// Default namespaces
	return []string{"infra", "app"}, nil
}

// GetScenarioValidation returns validation requirements for a scenario
func (m *Manager) GetScenarioValidation(scenario string) (requiredComponents []string, tests []string, err error) {
	config := m.GetConfig()

	sc, ok := config.Scenarios[scenario]
	if !ok {
		return nil, nil, fmt.Errorf("unknown scenario '%s'", scenario)
	}

	return sc.Validation.RequiredComponents, sc.Validation.Tests, nil
}

// ListAvailableScenarios returns all available scenarios with descriptions
func (m *Manager) ListAvailableScenarios() map[string]string {
	config := m.GetConfig()
	result := make(map[string]string)

	for name, sc := range config.Scenarios {
		result[name] = sc.Description
	}

	return result
}

// ListAvailableModules returns all available modules with descriptions
func (m *Manager) ListAvailableModules() map[string]string {
	config := m.GetConfig()
	result := make(map[string]string)

	for name, mod := range config.Modules {
		result[name] = mod.Description
	}

	return result
}

// ListAvailableOverlays returns all available overlays with descriptions
func (m *Manager) ListAvailableOverlays() map[string]string {
	config := m.GetConfig()
	result := make(map[string]string)

	for name, ov := range config.Overlays {
		result[name] = ov.Description
	}

	return result
}

// GetBlueprintMetadata returns blueprint metadata
func (m *Manager) GetBlueprintMetadata() Metadata {
	config := m.GetConfig()
	return config.Metadata
}

// GetBasePath returns the base path of the blueprint
func (m *Manager) GetBasePath() string {
	config := m.GetConfig()
	return config.Paths.Base
}

// Global manager instance
var (
	globalManager *Manager
	managerOnce   sync.Once
	managerErr    error
)

// GetGlobalManager returns the global blueprint manager instance
func GetGlobalManager() (*Manager, error) {
	managerOnce.Do(func() {
		// Try to find blueprint config
		configPath, err := FindBlueprintConfig(".")
		if err != nil {
			// Try some default locations
			defaultPaths := []string{
				"iaac/blueprint/.blueprint.yaml",
				"blueprint/.blueprint.yaml",
				".blueprint.yaml",
			}

			for _, path := range defaultPaths {
				if _, err := os.Stat(path); err == nil {
					configPath = path
					break
				}
			}

			if configPath == "" {
				managerErr = fmt.Errorf("blueprint configuration not found")
				return
			}
		}

		globalManager, managerErr = NewManager(configPath)
	})

	return globalManager, managerErr
}
