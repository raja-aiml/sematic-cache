package blueprint

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// Config represents the blueprint configuration
type Config struct {
	// Version of the blueprint config schema
	Version string `yaml:"version"`

	// Metadata about the blueprint
	Metadata Metadata `yaml:"metadata"`

	// Paths configuration
	Paths PathConfig `yaml:"paths"`

	// Scenarios available in this blueprint
	Scenarios map[string]ScenarioConfig `yaml:"scenarios"`

	// Modules available for composition
	Modules map[string]ModuleConfig `yaml:"modules,omitempty"`

	// Overlays available for customization
	Overlays map[string]OverlayConfig `yaml:"overlays,omitempty"`

	// Validation rules
	Validation ValidationConfig `yaml:"validation,omitempty"`
}

// Metadata contains blueprint metadata
type Metadata struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description,omitempty"`
	Version     string `yaml:"version,omitempty"`
	Author      string `yaml:"author,omitempty"`
}

// PathConfig defines the directory structure
type PathConfig struct {
	// Base path for the blueprint (relative to config file)
	Base string `yaml:"base,omitempty"`

	// Scenarios directory (relative to base)
	Scenarios string `yaml:"scenarios"`

	// Infrastructure directory (relative to base)
	Infrastructure string `yaml:"infrastructure"`

	// Application directory (relative to base)
	Application string `yaml:"application"`

	// Modules directory (relative to infrastructure)
	Modules string `yaml:"modules,omitempty"`

	// Overlays directory (relative to infrastructure)
	Overlays string `yaml:"overlays,omitempty"`

	// Validation kit directory (relative to base)
	ValidationKit string `yaml:"validation_kit,omitempty"`

	// Scripts/hack directory (relative to base)
	Scripts string `yaml:"scripts,omitempty"`
}

// ScenarioConfig defines a deployment scenario
type ScenarioConfig struct {
	Name        string   `yaml:"name"`
	Description string   `yaml:"description"`
	Path        string   `yaml:"path,omitempty"` // Override path if different from name
	Components  []string `yaml:"components"`     // List of modules/components included
	Overlays    []string `yaml:"overlays,omitempty"`
	Namespaces  []string `yaml:"namespaces,omitempty"`
	Validation  struct {
		RequiredComponents []string `yaml:"required_components,omitempty"`
		Tests              []string `yaml:"tests,omitempty"`
	} `yaml:"validation,omitempty"`
}

// ModuleConfig defines a reusable module
type ModuleConfig struct {
	Name         string            `yaml:"name"`
	Description  string            `yaml:"description,omitempty"`
	Path         string            `yaml:"path,omitempty"`
	Dependencies []string          `yaml:"dependencies,omitempty"`
	Parameters   map[string]string `yaml:"parameters,omitempty"`
}

// OverlayConfig defines an overlay configuration
type OverlayConfig struct {
	Name        string            `yaml:"name"`
	Description string            `yaml:"description,omitempty"`
	Path        string            `yaml:"path,omitempty"`
	Parameters  map[string]string `yaml:"parameters,omitempty"`
}

// ValidationConfig defines validation rules
type ValidationConfig struct {
	RequiredDirectories []string          `yaml:"required_directories,omitempty"`
	OptionalDirectories []string          `yaml:"optional_directories,omitempty"`
	RequiredFiles       []string          `yaml:"required_files,omitempty"`
	CustomRules         map[string]string `yaml:"custom_rules,omitempty"`
}

// LoadConfig loads blueprint configuration from a file
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config Config
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Set defaults
	config.setDefaults()

	// Resolve paths relative to config file
	configDir := filepath.Dir(path)
	config.resolvePaths(configDir)

	return &config, nil
}

// setDefaults sets default values for optional fields
func (c *Config) setDefaults() {
	if c.Version == "" {
		c.Version = "1.0"
	}

	if c.Paths.Base == "" {
		c.Paths.Base = "."
	}

	// Default paths if not specified
	if c.Paths.Scenarios == "" {
		c.Paths.Scenarios = "scenarios"
	}
	if c.Paths.Infrastructure == "" {
		c.Paths.Infrastructure = "infra"
	}
	if c.Paths.Application == "" {
		c.Paths.Application = "app"
	}
	if c.Paths.Modules == "" {
		c.Paths.Modules = "modules"
	}
	if c.Paths.Overlays == "" {
		c.Paths.Overlays = "overlays"
	}
}

// resolvePaths resolves all paths relative to the config directory
func (c *Config) resolvePaths(baseDir string) {
	// Resolve base path
	c.Paths.Base = filepath.Join(baseDir, c.Paths.Base)

	// Update scenario paths
	for name, scenario := range c.Scenarios {
		if scenario.Path == "" {
			scenario.Path = name
		}
		c.Scenarios[name] = scenario
	}

	// Update module paths
	for name, module := range c.Modules {
		if module.Path == "" {
			module.Path = name
		}
		c.Modules[name] = module
	}

	// Update overlay paths
	for name, overlay := range c.Overlays {
		if overlay.Path == "" {
			overlay.Path = name
		}
		c.Overlays[name] = overlay
	}
}

// GetScenarioPath returns the full path to a scenario
func (c *Config) GetScenarioPath(scenario string) string {
	if sc, ok := c.Scenarios[scenario]; ok && sc.Path != "" {
		return filepath.Join(c.Paths.Base, c.Paths.Scenarios, sc.Path)
	}
	return filepath.Join(c.Paths.Base, c.Paths.Scenarios, scenario)
}

// GetModulePath returns the full path to a module
func (c *Config) GetModulePath(module string) string {
	if mod, ok := c.Modules[module]; ok && mod.Path != "" {
		return filepath.Join(c.Paths.Base, c.Paths.Infrastructure, c.Paths.Modules, mod.Path)
	}
	return filepath.Join(c.Paths.Base, c.Paths.Infrastructure, c.Paths.Modules, module)
}

// GetOverlayPath returns the full path to an overlay
func (c *Config) GetOverlayPath(overlay string) string {
	if ov, ok := c.Overlays[overlay]; ok && ov.Path != "" {
		return filepath.Join(c.Paths.Base, c.Paths.Infrastructure, c.Paths.Overlays, ov.Path)
	}
	return filepath.Join(c.Paths.Base, c.Paths.Infrastructure, c.Paths.Overlays, overlay)
}

// GetInfraPath returns the infrastructure path
func (c *Config) GetInfraPath() string {
	return filepath.Join(c.Paths.Base, c.Paths.Infrastructure)
}

// GetAppPath returns the application path
func (c *Config) GetAppPath() string {
	return filepath.Join(c.Paths.Base, c.Paths.Application)
}

// ListScenarios returns a list of available scenario names
func (c *Config) ListScenarios() []string {
	scenarios := make([]string, 0, len(c.Scenarios))
	for name := range c.Scenarios {
		scenarios = append(scenarios, name)
	}
	return scenarios
}

// ListModules returns a list of available module names
func (c *Config) ListModules() []string {
	modules := make([]string, 0, len(c.Modules))
	for name := range c.Modules {
		modules = append(modules, name)
	}
	return modules
}

// ListOverlays returns a list of available overlay names
func (c *Config) ListOverlays() []string {
	overlays := make([]string, 0, len(c.Overlays))
	for name := range c.Overlays {
		overlays = append(overlays, name)
	}
	return overlays
}

// ValidateScenario checks if a scenario exists
func (c *Config) ValidateScenario(scenario string) error {
	if _, ok := c.Scenarios[scenario]; !ok {
		return fmt.Errorf("unknown scenario '%s', available: %v", scenario, c.ListScenarios())
	}
	return nil
}

// GetScenarioNamespaces returns expected namespaces for a scenario
func (c *Config) GetScenarioNamespaces(scenario string) []string {
	if sc, ok := c.Scenarios[scenario]; ok && len(sc.Namespaces) > 0 {
		return sc.Namespaces
	}
	// Default namespaces
	return []string{"infra", "app"}
}
