package blueprint

import (
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewManager(t *testing.T) {
	tests := []struct {
		name    string
		setup   func() (string, func())
		wantErr bool
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
scenarios:
  minimal:
    name: minimal
    description: Minimal scenario
`
				err := os.WriteFile(configPath, []byte(config), 0644)
				require.NoError(t, err)

				return configPath, func() {}
			},
			wantErr: false,
		},
		{
			name: "invalid_config_file",
			setup: func() (string, func()) {
				return "/non/existent/file.yaml", func() {}
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path, cleanup := tt.setup()
			defer cleanup()

			manager, err := NewManager(path)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, manager)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, manager)
				assert.NotNil(t, manager.config)
				assert.Equal(t, path, manager.configPath)
			}
		})
	}
}

func TestFindBlueprintConfig(t *testing.T) {
	tests := []struct {
		name    string
		setup   func() (string, func())
		wantErr bool
	}{
		{
			name: "config_in_current_directory",
			setup: func() (string, func()) {
				tmpDir := t.TempDir()
				configPath := filepath.Join(tmpDir, ".blueprint.yaml")

				err := os.WriteFile(configPath, []byte("version: 1.0"), 0644)
				require.NoError(t, err)

				return tmpDir, func() {}
			},
			wantErr: false,
		},
		{
			name: "config_in_parent_directory",
			setup: func() (string, func()) {
				tmpDir := t.TempDir()
				subDir := filepath.Join(tmpDir, "subdir")
				err := os.Mkdir(subDir, 0755)
				require.NoError(t, err)

				configPath := filepath.Join(tmpDir, "blueprint.yaml")
				err = os.WriteFile(configPath, []byte("version: 1.0"), 0644)
				require.NoError(t, err)

				return subDir, func() {}
			},
			wantErr: false,
		},
		{
			name: "config_in_iaac_blueprint_subdirectory",
			setup: func() (string, func()) {
				tmpDir := t.TempDir()
				iaacDir := filepath.Join(tmpDir, "iaac", "blueprint")
				err := os.MkdirAll(iaacDir, 0755)
				require.NoError(t, err)

				configPath := filepath.Join(iaacDir, ".blueprint.yml")
				err = os.WriteFile(configPath, []byte("version: 1.0"), 0644)
				require.NoError(t, err)

				return tmpDir, func() {}
			},
			wantErr: false,
		},
		{
			name: "config_in_blueprint_subdirectory",
			setup: func() (string, func()) {
				tmpDir := t.TempDir()
				blueprintDir := filepath.Join(tmpDir, "blueprint")
				err := os.MkdirAll(blueprintDir, 0755)
				require.NoError(t, err)

				configPath := filepath.Join(blueprintDir, "blueprint-config.yaml")
				err = os.WriteFile(configPath, []byte("version: 1.0"), 0644)
				require.NoError(t, err)

				return tmpDir, func() {}
			},
			wantErr: false,
		},
		{
			name: "no_config_found",
			setup: func() (string, func()) {
				tmpDir := t.TempDir()
				return tmpDir, func() {}
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startPath, cleanup := tt.setup()
			defer cleanup()

			configPath, err := FindBlueprintConfig(startPath)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Empty(t, configPath)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, configPath)
				assert.True(t, filepath.IsAbs(configPath))

				// Verify file exists
				_, err := os.Stat(configPath)
				assert.NoError(t, err)
			}
		})
	}
}

func TestManager_GetConfig(t *testing.T) {
	config := &Config{
		Version: "1.0",
		Metadata: Metadata{
			Name: "test-blueprint",
		},
	}

	manager := &Manager{
		config: config,
	}

	got := manager.GetConfig()
	assert.Equal(t, config, got)
}

func TestManager_ReloadConfig(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "blueprint.yaml")

	// Write initial config
	initialConfig := `
version: "1.0"
metadata:
  name: initial-blueprint
scenarios:
  minimal:
    name: minimal
`
	err := os.WriteFile(configPath, []byte(initialConfig), 0644)
	require.NoError(t, err)

	manager, err := NewManager(configPath)
	require.NoError(t, err)

	// Verify initial config
	assert.Equal(t, "initial-blueprint", manager.GetConfig().Metadata.Name)

	// Update config file
	updatedConfig := `
version: "1.0"
metadata:
  name: updated-blueprint
scenarios:
  minimal:
    name: minimal
  development:
    name: development
`
	err = os.WriteFile(configPath, []byte(updatedConfig), 0644)
	require.NoError(t, err)

	// Reload config
	err = manager.ReloadConfig()
	require.NoError(t, err)

	// Verify updated config
	assert.Equal(t, "updated-blueprint", manager.GetConfig().Metadata.Name)
	assert.Len(t, manager.GetConfig().Scenarios, 2)
}

func TestManager_ValidateStructure(t *testing.T) {
	tests := []struct {
		name    string
		setup   func() *Manager
		wantErr bool
	}{
		{
			name: "valid_structure",
			setup: func() *Manager {
				tmpDir := t.TempDir()

				// Create required directories
				err := os.MkdirAll(filepath.Join(tmpDir, "scenarios"), 0755)
				require.NoError(t, err)
				err = os.MkdirAll(filepath.Join(tmpDir, "infra"), 0755)
				require.NoError(t, err)

				// Create required files
				err = os.WriteFile(filepath.Join(tmpDir, "README.md"), []byte("# README"), 0644)
				require.NoError(t, err)

				return &Manager{
					config: &Config{
						Paths: PathConfig{
							Base: tmpDir,
						},
						Validation: ValidationConfig{
							RequiredDirectories: []string{"scenarios", "infra"},
							RequiredFiles:       []string{"README.md"},
						},
					},
				}
			},
			wantErr: false,
		},
		{
			name: "missing_required_directory",
			setup: func() *Manager {
				tmpDir := t.TempDir()

				return &Manager{
					config: &Config{
						Paths: PathConfig{
							Base: tmpDir,
						},
						Validation: ValidationConfig{
							RequiredDirectories: []string{"scenarios", "missing-dir"},
						},
					},
				}
			},
			wantErr: true,
		},
		{
			name: "missing_required_file",
			setup: func() *Manager {
				tmpDir := t.TempDir()

				return &Manager{
					config: &Config{
						Paths: PathConfig{
							Base: tmpDir,
						},
						Validation: ValidationConfig{
							RequiredFiles: []string{"missing-file.txt"},
						},
					},
				}
			},
			wantErr: true,
		},
		{
			name: "no_validation_rules",
			setup: func() *Manager {
				return &Manager{
					config: &Config{
						Paths: PathConfig{
							Base: t.TempDir(),
						},
						Validation: ValidationConfig{},
					},
				}
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manager := tt.setup()
			err := manager.ValidateStructure()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestManager_GetScenarioPath(t *testing.T) {
	manager := &Manager{
		config: &Config{
			Paths: PathConfig{
				Base:      "/project/blueprint",
				Scenarios: "scenarios",
			},
			Scenarios: map[string]ScenarioConfig{
				"minimal": {
					Name: "minimal",
					Path: "minimal",
				},
			},
		},
	}

	tests := []struct {
		name     string
		scenario string
		want     string
		wantErr  bool
	}{
		{
			name:     "valid_scenario",
			scenario: "minimal",
			want:     filepath.Join("/project/blueprint", "scenarios", "minimal"),
			wantErr:  false,
		},
		{
			name:     "invalid_scenario",
			scenario: "unknown",
			want:     "",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := manager.GetScenarioPath(tt.scenario)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestManager_GetModulePath(t *testing.T) {
	manager := &Manager{
		config: &Config{
			Paths: PathConfig{
				Base:           "/project/blueprint",
				Infrastructure: "infra",
				Modules:        "modules",
			},
			Modules: map[string]ModuleConfig{
				"postgres": {
					Name: "postgres",
					Path: "postgres",
				},
			},
		},
	}

	tests := []struct {
		name    string
		module  string
		want    string
		wantErr bool
	}{
		{
			name:    "valid_module",
			module:  "postgres",
			want:    filepath.Join("/project/blueprint", "infra", "modules", "postgres"),
			wantErr: false,
		},
		{
			name:    "invalid_module",
			module:  "unknown",
			want:    "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := manager.GetModulePath(tt.module)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestManager_GetOverlayPath(t *testing.T) {
	manager := &Manager{
		config: &Config{
			Paths: PathConfig{
				Base:           "/project/blueprint",
				Infrastructure: "infra",
				Overlays:       "overlays",
			},
			Overlays: map[string]OverlayConfig{
				"production": {
					Name: "production",
					Path: "production",
				},
			},
		},
	}

	tests := []struct {
		name    string
		overlay string
		want    string
		wantErr bool
	}{
		{
			name:    "valid_overlay",
			overlay: "production",
			want:    filepath.Join("/project/blueprint", "infra", "overlays", "production"),
			wantErr: false,
		},
		{
			name:    "invalid_overlay",
			overlay: "unknown",
			want:    "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := manager.GetOverlayPath(tt.overlay)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestManager_GetScenarioComponents(t *testing.T) {
	manager := &Manager{
		config: &Config{
			Scenarios: map[string]ScenarioConfig{
				"minimal": {
					Name:       "minimal",
					Components: []string{"postgres", "redis"},
				},
			},
		},
	}

	tests := []struct {
		name     string
		scenario string
		want     []string
		wantErr  bool
	}{
		{
			name:     "valid_scenario",
			scenario: "minimal",
			want:     []string{"postgres", "redis"},
			wantErr:  false,
		},
		{
			name:     "invalid_scenario",
			scenario: "unknown",
			want:     nil,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := manager.GetScenarioComponents(tt.scenario)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestManager_GetScenarioNamespaces(t *testing.T) {
	manager := &Manager{
		config: &Config{
			Scenarios: map[string]ScenarioConfig{
				"minimal": {
					Name:       "minimal",
					Namespaces: []string{"infra", "app", "custom"},
				},
				"development": {
					Name: "development",
					// No namespaces specified
				},
			},
		},
	}

	tests := []struct {
		name     string
		scenario string
		want     []string
		wantErr  bool
	}{
		{
			name:     "scenario_with_namespaces",
			scenario: "minimal",
			want:     []string{"infra", "app", "custom"},
			wantErr:  false,
		},
		{
			name:     "scenario_without_namespaces",
			scenario: "development",
			want:     []string{"infra", "app"}, // Default namespaces
			wantErr:  false,
		},
		{
			name:     "invalid_scenario",
			scenario: "unknown",
			want:     nil,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := manager.GetScenarioNamespaces(tt.scenario)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestManager_GetScenarioValidation(t *testing.T) {
	manager := &Manager{
		config: &Config{
			Scenarios: map[string]ScenarioConfig{
				"minimal": {
					Name: "minimal",
					Validation: struct {
						RequiredComponents []string `yaml:"required_components,omitempty"`
						Tests              []string `yaml:"tests,omitempty"`
					}{
						RequiredComponents: []string{"postgres", "redis"},
						Tests:              []string{"connectivity", "health"},
					},
				},
			},
		},
	}

	tests := []struct {
		name           string
		scenario       string
		wantComponents []string
		wantTests      []string
		wantErr        bool
	}{
		{
			name:           "valid_scenario",
			scenario:       "minimal",
			wantComponents: []string{"postgres", "redis"},
			wantTests:      []string{"connectivity", "health"},
			wantErr:        false,
		},
		{
			name:           "invalid_scenario",
			scenario:       "unknown",
			wantComponents: nil,
			wantTests:      nil,
			wantErr:        true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			components, tests, err := manager.GetScenarioValidation(tt.scenario)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.wantComponents, components)
				assert.Equal(t, tt.wantTests, tests)
			}
		})
	}
}

func TestManager_ListAvailableScenarios(t *testing.T) {
	manager := &Manager{
		config: &Config{
			Scenarios: map[string]ScenarioConfig{
				"minimal": {
					Name:        "minimal",
					Description: "Minimal deployment",
				},
				"development": {
					Name:        "development",
					Description: "Development environment",
				},
			},
		},
	}

	got := manager.ListAvailableScenarios()
	assert.Len(t, got, 2)
	assert.Equal(t, "Minimal deployment", got["minimal"])
	assert.Equal(t, "Development environment", got["development"])
}

func TestManager_ListAvailableModules(t *testing.T) {
	manager := &Manager{
		config: &Config{
			Modules: map[string]ModuleConfig{
				"postgres": {
					Name:        "postgres",
					Description: "PostgreSQL database",
				},
				"redis": {
					Name:        "redis",
					Description: "Redis cache",
				},
			},
		},
	}

	got := manager.ListAvailableModules()
	assert.Len(t, got, 2)
	assert.Equal(t, "PostgreSQL database", got["postgres"])
	assert.Equal(t, "Redis cache", got["redis"])
}

func TestManager_ListAvailableOverlays(t *testing.T) {
	manager := &Manager{
		config: &Config{
			Overlays: map[string]OverlayConfig{
				"production": {
					Name:        "production",
					Description: "Production settings",
				},
				"development": {
					Name:        "development",
					Description: "Development settings",
				},
			},
		},
	}

	got := manager.ListAvailableOverlays()
	assert.Len(t, got, 2)
	assert.Equal(t, "Production settings", got["production"])
	assert.Equal(t, "Development settings", got["development"])
}

func TestManager_GetBlueprintMetadata(t *testing.T) {
	metadata := Metadata{
		Name:        "test-blueprint",
		Description: "Test blueprint",
		Version:     "1.0.0",
		Author:      "Test Author",
	}

	manager := &Manager{
		config: &Config{
			Metadata: metadata,
		},
	}

	got := manager.GetBlueprintMetadata()
	assert.Equal(t, metadata, got)
}

func TestManager_GetBasePath(t *testing.T) {
	manager := &Manager{
		config: &Config{
			Paths: PathConfig{
				Base: "/project/blueprint",
			},
		},
	}

	got := manager.GetBasePath()
	assert.Equal(t, "/project/blueprint", got)
}

func TestGetGlobalManager(t *testing.T) {
	// Reset singleton for test
	managerOnce = sync.Once{}
	globalManager = nil
	managerErr = nil

	tests := []struct {
		name    string
		setup   func() func()
		wantErr bool
	}{
		{
			name: "finds_config_in_current_directory",
			setup: func() func() {
				// Since there's already a config in the project, we'll test with that
				return func() {}
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Reset singleton for each test
			managerOnce = sync.Once{}
			globalManager = nil
			managerErr = nil

			cleanup := tt.setup()
			defer cleanup()

			manager, err := GetGlobalManager()
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, manager)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, manager)
			}

			// Test singleton behavior
			manager2, err2 := GetGlobalManager()
			assert.Same(t, manager, manager2)
			assert.Equal(t, err, err2)
		})
	}
}

func TestManager_ConcurrentAccess(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "blueprint.yaml")

	config := `
version: "1.0"
metadata:
  name: concurrent-test
scenarios:
  minimal:
    name: minimal
    namespaces: ["infra", "app"]
`
	err := os.WriteFile(configPath, []byte(config), 0644)
	require.NoError(t, err)

	manager, err := NewManager(configPath)
	require.NoError(t, err)

	// Test concurrent access
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			// Concurrent reads
			_ = manager.GetConfig()
			_ = manager.GetBasePath()
			_ = manager.GetBlueprintMetadata()
			_, _ = manager.GetScenarioPath("minimal")
			_, _ = manager.GetScenarioNamespaces("minimal")

			// Concurrent reload (write operation)
			if i%10 == 0 {
				_ = manager.ReloadConfig()
			}
		}()
	}
	wg.Wait()

	// Should complete without race conditions
	assert.NotNil(t, manager)
}
