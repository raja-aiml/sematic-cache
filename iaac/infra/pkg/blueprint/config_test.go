package blueprint

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestLoadConfig(t *testing.T) {
	tests := []struct {
		name    string
		setup   func() (string, func())
		wantErr bool
		check   func(*testing.T, *Config)
	}{
		{
			name: "valid_config_file",
			setup: func() (string, func()) {
				tmpDir := t.TempDir()
				configPath := filepath.Join(tmpDir, "blueprint.yaml")

				config := `
version: "1.0"
metadata:
  name: test-blueprint
  description: Test blueprint for unit tests
paths:
  scenarios: scenarios
  infrastructure: infra
  application: app
scenarios:
  minimal:
    name: minimal
    description: Minimal scenario
    components: ["postgres", "redis"]
    namespaces: ["infra", "app"]
modules:
  postgres:
    name: postgres
    description: PostgreSQL module
  redis:
    name: redis
    description: Redis module
`
				err := os.WriteFile(configPath, []byte(config), 0644)
				require.NoError(t, err)

				return configPath, func() {}
			},
			wantErr: false,
			check: func(t *testing.T, c *Config) {
				assert.Equal(t, "1.0", c.Version)
				assert.Equal(t, "test-blueprint", c.Metadata.Name)
				assert.Equal(t, "Test blueprint for unit tests", c.Metadata.Description)
				assert.Equal(t, "scenarios", c.Paths.Scenarios)
				assert.Equal(t, "infra", c.Paths.Infrastructure)
				assert.Equal(t, "app", c.Paths.Application)
				assert.Len(t, c.Scenarios, 1)
				assert.Len(t, c.Modules, 2)
			},
		},
		{
			name: "file_not_found",
			setup: func() (string, func()) {
				return "/non/existent/file.yaml", func() {}
			},
			wantErr: true,
		},
		{
			name: "invalid_yaml",
			setup: func() (string, func()) {
				tmpDir := t.TempDir()
				configPath := filepath.Join(tmpDir, "blueprint.yaml")

				err := os.WriteFile(configPath, []byte("invalid: yaml: content:"), 0644)
				require.NoError(t, err)

				return configPath, func() {}
			},
			wantErr: true,
		},
		{
			name: "config_with_defaults",
			setup: func() (string, func()) {
				tmpDir := t.TempDir()
				configPath := filepath.Join(tmpDir, "blueprint.yaml")

				config := `
metadata:
  name: minimal-blueprint
scenarios:
  test:
    name: test
    description: Test scenario
`
				err := os.WriteFile(configPath, []byte(config), 0644)
				require.NoError(t, err)

				return configPath, func() {}
			},
			wantErr: false,
			check: func(t *testing.T, c *Config) {
				assert.Equal(t, "1.0", c.Version)                // Default version
				assert.Equal(t, "scenarios", c.Paths.Scenarios)  // Default path
				assert.Equal(t, "infra", c.Paths.Infrastructure) // Default path
				assert.Equal(t, "app", c.Paths.Application)      // Default path
				assert.Equal(t, "modules", c.Paths.Modules)      // Default path
				assert.Equal(t, "overlays", c.Paths.Overlays)    // Default path
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path, cleanup := tt.setup()
			defer cleanup()

			config, err := LoadConfig(path)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, config)

			if tt.check != nil {
				tt.check(t, config)
			}
		})
	}
}

func TestConfig_setDefaults(t *testing.T) {
	config := &Config{}
	config.setDefaults()

	assert.Equal(t, "1.0", config.Version)
	assert.Equal(t, ".", config.Paths.Base)
	assert.Equal(t, "scenarios", config.Paths.Scenarios)
	assert.Equal(t, "infra", config.Paths.Infrastructure)
	assert.Equal(t, "app", config.Paths.Application)
	assert.Equal(t, "modules", config.Paths.Modules)
	assert.Equal(t, "overlays", config.Paths.Overlays)
}

func TestConfig_resolvePaths(t *testing.T) {
	tests := []struct {
		name    string
		config  *Config
		baseDir string
		check   func(*testing.T, *Config)
	}{
		{
			name: "resolve_base_path",
			config: &Config{
				Paths: PathConfig{
					Base: "blueprint",
				},
			},
			baseDir: "/project",
			check: func(t *testing.T, c *Config) {
				assert.Equal(t, filepath.Join("/project", "blueprint"), c.Paths.Base)
			},
		},
		{
			name: "resolve_scenario_paths",
			config: &Config{
				Paths: PathConfig{
					Base: ".",
				},
				Scenarios: map[string]ScenarioConfig{
					"test1": {Name: "test1"},
					"test2": {Name: "test2", Path: "custom-path"},
				},
			},
			baseDir: "/project",
			check: func(t *testing.T, c *Config) {
				assert.Equal(t, "test1", c.Scenarios["test1"].Path)
				assert.Equal(t, "custom-path", c.Scenarios["test2"].Path)
			},
		},
		{
			name: "resolve_module_paths",
			config: &Config{
				Paths: PathConfig{
					Base: ".",
				},
				Modules: map[string]ModuleConfig{
					"mod1": {Name: "mod1"},
					"mod2": {Name: "mod2", Path: "custom-mod"},
				},
			},
			baseDir: "/project",
			check: func(t *testing.T, c *Config) {
				assert.Equal(t, "mod1", c.Modules["mod1"].Path)
				assert.Equal(t, "custom-mod", c.Modules["mod2"].Path)
			},
		},
		{
			name: "resolve_overlay_paths",
			config: &Config{
				Paths: PathConfig{
					Base: ".",
				},
				Overlays: map[string]OverlayConfig{
					"ov1": {Name: "ov1"},
					"ov2": {Name: "ov2", Path: "custom-overlay"},
				},
			},
			baseDir: "/project",
			check: func(t *testing.T, c *Config) {
				assert.Equal(t, "ov1", c.Overlays["ov1"].Path)
				assert.Equal(t, "custom-overlay", c.Overlays["ov2"].Path)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.config.resolvePaths(tt.baseDir)
			tt.check(t, tt.config)
		})
	}
}

func TestConfig_GetScenarioPath(t *testing.T) {
	config := &Config{
		Paths: PathConfig{
			Base:      "/project/blueprint",
			Scenarios: "scenarios",
		},
		Scenarios: map[string]ScenarioConfig{
			"minimal": {
				Name: "minimal",
				Path: "minimal-custom",
			},
			"development": {
				Name: "development",
			},
		},
	}

	tests := []struct {
		name     string
		scenario string
		want     string
	}{
		{
			name:     "scenario_with_custom_path",
			scenario: "minimal",
			want:     filepath.Join("/project/blueprint", "scenarios", "minimal-custom"),
		},
		{
			name:     "scenario_with_default_path",
			scenario: "development",
			want:     filepath.Join("/project/blueprint", "scenarios", "development"),
		},
		{
			name:     "unknown_scenario",
			scenario: "unknown",
			want:     filepath.Join("/project/blueprint", "scenarios", "unknown"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := config.GetScenarioPath(tt.scenario)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestConfig_GetModulePath(t *testing.T) {
	config := &Config{
		Paths: PathConfig{
			Base:           "/project/blueprint",
			Infrastructure: "infra",
			Modules:        "modules",
		},
		Modules: map[string]ModuleConfig{
			"postgres": {
				Name: "postgres",
				Path: "database/postgres",
			},
			"redis": {
				Name: "redis",
			},
		},
	}

	tests := []struct {
		name   string
		module string
		want   string
	}{
		{
			name:   "module_with_custom_path",
			module: "postgres",
			want:   filepath.Join("/project/blueprint", "infra", "modules", "database/postgres"),
		},
		{
			name:   "module_with_default_path",
			module: "redis",
			want:   filepath.Join("/project/blueprint", "infra", "modules", "redis"),
		},
		{
			name:   "unknown_module",
			module: "unknown",
			want:   filepath.Join("/project/blueprint", "infra", "modules", "unknown"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := config.GetModulePath(tt.module)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestConfig_GetOverlayPath(t *testing.T) {
	config := &Config{
		Paths: PathConfig{
			Base:           "/project/blueprint",
			Infrastructure: "infra",
			Overlays:       "overlays",
		},
		Overlays: map[string]OverlayConfig{
			"production": {
				Name: "production",
				Path: "env/production",
			},
			"development": {
				Name: "development",
			},
		},
	}

	tests := []struct {
		name    string
		overlay string
		want    string
	}{
		{
			name:    "overlay_with_custom_path",
			overlay: "production",
			want:    filepath.Join("/project/blueprint", "infra", "overlays", "env/production"),
		},
		{
			name:    "overlay_with_default_path",
			overlay: "development",
			want:    filepath.Join("/project/blueprint", "infra", "overlays", "development"),
		},
		{
			name:    "unknown_overlay",
			overlay: "unknown",
			want:    filepath.Join("/project/blueprint", "infra", "overlays", "unknown"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := config.GetOverlayPath(tt.overlay)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestConfig_GetInfraPath(t *testing.T) {
	config := &Config{
		Paths: PathConfig{
			Base:           "/project/blueprint",
			Infrastructure: "infrastructure",
		},
	}

	got := config.GetInfraPath()
	want := filepath.Join("/project/blueprint", "infrastructure")
	assert.Equal(t, want, got)
}

func TestConfig_GetAppPath(t *testing.T) {
	config := &Config{
		Paths: PathConfig{
			Base:        "/project/blueprint",
			Application: "application",
		},
	}

	got := config.GetAppPath()
	want := filepath.Join("/project/blueprint", "application")
	assert.Equal(t, want, got)
}

func TestConfig_ListScenarios(t *testing.T) {
	config := &Config{
		Scenarios: map[string]ScenarioConfig{
			"minimal":     {},
			"development": {},
			"production":  {},
		},
	}

	got := config.ListScenarios()
	assert.Len(t, got, 3)
	assert.Contains(t, got, "minimal")
	assert.Contains(t, got, "development")
	assert.Contains(t, got, "production")
}

func TestConfig_ListModules(t *testing.T) {
	config := &Config{
		Modules: map[string]ModuleConfig{
			"postgres": {},
			"redis":    {},
			"nginx":    {},
		},
	}

	got := config.ListModules()
	assert.Len(t, got, 3)
	assert.Contains(t, got, "postgres")
	assert.Contains(t, got, "redis")
	assert.Contains(t, got, "nginx")
}

func TestConfig_ListOverlays(t *testing.T) {
	config := &Config{
		Overlays: map[string]OverlayConfig{
			"production":  {},
			"development": {},
			"staging":     {},
		},
	}

	got := config.ListOverlays()
	assert.Len(t, got, 3)
	assert.Contains(t, got, "production")
	assert.Contains(t, got, "development")
	assert.Contains(t, got, "staging")
}

func TestConfig_ValidateScenario(t *testing.T) {
	config := &Config{
		Scenarios: map[string]ScenarioConfig{
			"minimal":     {},
			"development": {},
		},
	}

	tests := []struct {
		name     string
		scenario string
		wantErr  bool
	}{
		{
			name:     "valid_scenario",
			scenario: "minimal",
			wantErr:  false,
		},
		{
			name:     "another_valid_scenario",
			scenario: "development",
			wantErr:  false,
		},
		{
			name:     "invalid_scenario",
			scenario: "unknown",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := config.ValidateScenario(tt.scenario)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "unknown scenario")
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestConfig_GetScenarioNamespaces(t *testing.T) {
	config := &Config{
		Scenarios: map[string]ScenarioConfig{
			"minimal": {
				Namespaces: []string{"infra", "app", "custom"},
			},
			"development": {
				// No namespaces specified
			},
		},
	}

	tests := []struct {
		name     string
		scenario string
		want     []string
	}{
		{
			name:     "scenario_with_namespaces",
			scenario: "minimal",
			want:     []string{"infra", "app", "custom"},
		},
		{
			name:     "scenario_without_namespaces",
			scenario: "development",
			want:     []string{"infra", "app"}, // Default namespaces
		},
		{
			name:     "unknown_scenario",
			scenario: "unknown",
			want:     []string{"infra", "app"}, // Default namespaces
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := config.GetScenarioNamespaces(tt.scenario)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestConfig_YAML_Marshal_Unmarshal(t *testing.T) {
	original := &Config{
		Version: "1.0",
		Metadata: Metadata{
			Name:        "test-blueprint",
			Description: "Test blueprint",
			Version:     "0.1.0",
			Author:      "Test Author",
		},
		Paths: PathConfig{
			Base:           ".",
			Scenarios:      "scenarios",
			Infrastructure: "infra",
			Application:    "app",
			Modules:        "modules",
			Overlays:       "overlays",
			ValidationKit:  "validation",
			Scripts:        "scripts",
		},
		Scenarios: map[string]ScenarioConfig{
			"minimal": {
				Name:        "minimal",
				Description: "Minimal scenario",
				Path:        "minimal",
				Components:  []string{"postgres", "redis"},
				Overlays:    []string{"base"},
				Namespaces:  []string{"infra", "app"},
				Validation: struct {
					RequiredComponents []string `yaml:"required_components,omitempty"`
					Tests              []string `yaml:"tests,omitempty"`
				}{
					RequiredComponents: []string{"postgres"},
					Tests:              []string{"connectivity", "health"},
				},
			},
		},
		Modules: map[string]ModuleConfig{
			"postgres": {
				Name:         "postgres",
				Description:  "PostgreSQL database",
				Path:         "postgres",
				Dependencies: []string{"storage"},
				Parameters: map[string]string{
					"version": "14",
					"storage": "10Gi",
				},
			},
		},
		Overlays: map[string]OverlayConfig{
			"production": {
				Name:        "production",
				Description: "Production overlay",
				Path:        "production",
				Parameters: map[string]string{
					"replicas": "3",
					"cpu":      "2",
				},
			},
		},
		Validation: ValidationConfig{
			RequiredDirectories: []string{"scenarios", "infra"},
			OptionalDirectories: []string{"scripts"},
			RequiredFiles:       []string{"README.md"},
			CustomRules: map[string]string{
				"namespace_prefix": "All namespaces must start with 'app-'",
			},
		},
	}

	// Marshal to YAML
	data, err := yaml.Marshal(original)
	require.NoError(t, err)

	// Unmarshal back
	var restored Config
	err = yaml.Unmarshal(data, &restored)
	require.NoError(t, err)

	// Compare
	assert.Equal(t, original.Version, restored.Version)
	assert.Equal(t, original.Metadata, restored.Metadata)
	assert.Equal(t, original.Paths, restored.Paths)
	assert.Equal(t, original.Scenarios, restored.Scenarios)
	assert.Equal(t, original.Modules, restored.Modules)
	assert.Equal(t, original.Overlays, restored.Overlays)
	assert.Equal(t, original.Validation, restored.Validation)
}
