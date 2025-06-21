package blueprint

import (
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetAdapter(t *testing.T) {
	tests := []struct {
		name    string
		wantErr bool
	}{
		{
			name:    "singleton_returns_same_instance",
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Reset singleton for test
			adapterOnce = sync.Once{}
			globalAdapter = nil

			adapter1 := GetAdapter()
			adapter2 := GetAdapter()

			assert.Same(t, adapter1, adapter2, "GetAdapter should return the same instance")
			assert.NotNil(t, adapter1, "adapter should not be nil")
		})
	}
}

func TestAdapter_GetScenarioPath(t *testing.T) {
	tests := []struct {
		name     string
		scenario string
		setup    func(*Adapter)
		want     string
	}{
		{
			name:     "returns_path_from_manager",
			scenario: "minimal",
			setup: func(a *Adapter) {
				mockManager := &Manager{
					config: &Config{
						Paths: PathConfig{
							Base:      "/test/base",
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
				a.manager = mockManager
			},
			want: "/test/base/scenarios/minimal",
		},
		{
			name:     "fallback_to_hardcoded_path",
			scenario: "development",
			setup: func(a *Adapter) {
				a.manager = nil
			},
			want: filepath.Join("iaac", "blueprint", "scenarios", "development"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &Adapter{}
			if tt.setup != nil {
				tt.setup(adapter)
			}

			got := adapter.GetScenarioPath(tt.scenario)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestAdapter_GetModulePath(t *testing.T) {
	tests := []struct {
		name   string
		module string
		setup  func(*Adapter)
		want   string
	}{
		{
			name:   "returns_path_from_manager",
			module: "postgres",
			setup: func(a *Adapter) {
				mockManager := &Manager{
					config: &Config{
						Paths: PathConfig{
							Base:           "/test/base",
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
				a.manager = mockManager
			},
			want: "/test/base/infra/modules/postgres",
		},
		{
			name:   "fallback_to_hardcoded_path",
			module: "redis",
			setup: func(a *Adapter) {
				a.manager = nil
			},
			want: filepath.Join("iaac", "blueprint", "infra", "modules", "redis"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &Adapter{}
			if tt.setup != nil {
				tt.setup(adapter)
			}

			got := adapter.GetModulePath(tt.module)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestAdapter_GetOverlayPath(t *testing.T) {
	tests := []struct {
		name    string
		overlay string
		setup   func(*Adapter)
		want    string
	}{
		{
			name:    "returns_path_from_manager",
			overlay: "production",
			setup: func(a *Adapter) {
				mockManager := &Manager{
					config: &Config{
						Paths: PathConfig{
							Base:           "/test/base",
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
				a.manager = mockManager
			},
			want: "/test/base/infra/overlays/production",
		},
		{
			name:    "fallback_to_hardcoded_path",
			overlay: "development",
			setup: func(a *Adapter) {
				a.manager = nil
			},
			want: filepath.Join("iaac", "blueprint", "infra", "overlays", "development"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &Adapter{}
			if tt.setup != nil {
				tt.setup(adapter)
			}

			got := adapter.GetOverlayPath(tt.overlay)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestAdapter_GetScenarioNamespaces(t *testing.T) {
	tests := []struct {
		name     string
		scenario string
		setup    func(*Adapter)
		want     []string
	}{
		{
			name:     "returns_namespaces_from_manager",
			scenario: "minimal",
			setup: func(a *Adapter) {
				mockManager := &Manager{
					config: &Config{
						Scenarios: map[string]ScenarioConfig{
							"minimal": {
								Name:       "minimal",
								Namespaces: []string{"infra", "app", "custom"},
							},
						},
					},
				}
				a.manager = mockManager
			},
			want: []string{"infra", "app", "custom"},
		},
		{
			name:     "fallback_minimal_scenario",
			scenario: "minimal",
			setup: func(a *Adapter) {
				a.manager = nil
			},
			want: []string{"infra", "app"},
		},
		{
			name:     "fallback_development_scenario",
			scenario: "development",
			setup: func(a *Adapter) {
				a.manager = nil
			},
			want: []string{"infra", "app", "dev-tools"},
		},
		{
			name:     "fallback_service_mesh_scenario",
			scenario: "service-mesh",
			setup: func(a *Adapter) {
				a.manager = nil
			},
			want: []string{"infra", "app", "istio-system", "istio-ingress"},
		},
		{
			name:     "fallback_monitoring_only_scenario",
			scenario: "monitoring-only",
			setup: func(a *Adapter) {
				a.manager = nil
			},
			want: []string{"infra", "app", "monitoring", "logging"},
		},
		{
			name:     "fallback_full_stack_scenario",
			scenario: "full-stack",
			setup: func(a *Adapter) {
				a.manager = nil
			},
			want: []string{"infra", "app", "istio-system", "istio-ingress", "monitoring", "logging", "dev-tools"},
		},
		{
			name:     "fallback_unknown_scenario",
			scenario: "unknown",
			setup: func(a *Adapter) {
				a.manager = nil
			},
			want: []string{"infra", "app"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &Adapter{}
			if tt.setup != nil {
				tt.setup(adapter)
			}

			got := adapter.GetScenarioNamespaces(tt.scenario)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestAdapter_ListScenarios(t *testing.T) {
	tests := []struct {
		name  string
		setup func(*Adapter)
		want  []string
	}{
		{
			name: "returns_scenarios_from_manager",
			setup: func(a *Adapter) {
				mockManager := &Manager{
					config: &Config{
						Scenarios: map[string]ScenarioConfig{
							"minimal":     {},
							"development": {},
							"production":  {},
						},
					},
				}
				a.manager = mockManager
			},
			want: []string{"minimal", "development", "production"},
		},
		{
			name: "fallback_to_hardcoded_list",
			setup: func(a *Adapter) {
				a.manager = nil
			},
			want: []string{
				"minimal",
				"development",
				"service-mesh",
				"monitoring-only",
				"full-stack",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &Adapter{}
			if tt.setup != nil {
				tt.setup(adapter)
			}

			got := adapter.ListScenarios()
			if adapter.manager != nil {
				// When using manager, order might vary
				assert.ElementsMatch(t, tt.want, got)
			} else {
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestAdapter_ValidateScenario(t *testing.T) {
	tests := []struct {
		name     string
		scenario string
		setup    func(*Adapter)
		wantErr  bool
	}{
		{
			name:     "valid_scenario_from_manager",
			scenario: "minimal",
			setup: func(a *Adapter) {
				mockManager := &Manager{
					config: &Config{
						Scenarios: map[string]ScenarioConfig{
							"minimal": {},
						},
					},
				}
				a.manager = mockManager
			},
			wantErr: false,
		},
		{
			name:     "invalid_scenario_from_manager",
			scenario: "unknown",
			setup: func(a *Adapter) {
				mockManager := &Manager{
					config: &Config{
						Scenarios: map[string]ScenarioConfig{
							"minimal": {},
						},
					},
				}
				a.manager = mockManager
			},
			wantErr: true,
		},
		{
			name:     "valid_scenario_fallback",
			scenario: "minimal",
			setup: func(a *Adapter) {
				a.manager = nil
			},
			wantErr: false,
		},
		{
			name:     "invalid_scenario_fallback",
			scenario: "unknown",
			setup: func(a *Adapter) {
				a.manager = nil
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &Adapter{}
			if tt.setup != nil {
				tt.setup(adapter)
			}

			err := adapter.ValidateScenario(tt.scenario)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestAdapter_GetBlueprintPath(t *testing.T) {
	tests := []struct {
		name      string
		component string
		setup     func(*Adapter)
		want      string
	}{
		{
			name:      "returns_path_from_manager",
			component: "scripts",
			setup: func(a *Adapter) {
				mockManager := &Manager{
					config: &Config{
						Paths: PathConfig{
							Base: "/test/blueprint",
						},
					},
				}
				a.manager = mockManager
			},
			want: filepath.Join("/test/blueprint", "scripts"),
		},
		{
			name:      "fallback_to_hardcoded_path",
			component: "scripts",
			setup: func(a *Adapter) {
				a.manager = nil
			},
			want: filepath.Join("iaac", "blueprint", "scripts"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &Adapter{}
			if tt.setup != nil {
				tt.setup(adapter)
			}

			got := adapter.GetBlueprintPath(tt.component)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestAdapter_GetInfraPath(t *testing.T) {
	tests := []struct {
		name  string
		setup func(*Adapter)
		want  string
	}{
		{
			name: "returns_path_from_manager",
			setup: func(a *Adapter) {
				mockManager := &Manager{
					config: &Config{
						Paths: PathConfig{
							Base:           "/test/blueprint",
							Infrastructure: "infrastructure",
						},
					},
				}
				a.manager = mockManager
			},
			want: filepath.Join("/test/blueprint", "infrastructure"),
		},
		{
			name: "fallback_to_hardcoded_path",
			setup: func(a *Adapter) {
				a.manager = nil
			},
			want: filepath.Join("iaac", "blueprint", "infra"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &Adapter{}
			if tt.setup != nil {
				tt.setup(adapter)
			}

			got := adapter.GetInfraPath()
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestAdapter_GetAppPath(t *testing.T) {
	tests := []struct {
		name  string
		setup func(*Adapter)
		want  string
	}{
		{
			name: "returns_path_from_manager",
			setup: func(a *Adapter) {
				mockManager := &Manager{
					config: &Config{
						Paths: PathConfig{
							Base:        "/test/blueprint",
							Application: "application",
						},
					},
				}
				a.manager = mockManager
			},
			want: filepath.Join("/test/blueprint", "application"),
		},
		{
			name: "fallback_to_hardcoded_path",
			setup: func(a *Adapter) {
				a.manager = nil
			},
			want: filepath.Join("iaac", "blueprint", "app"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &Adapter{}
			if tt.setup != nil {
				tt.setup(adapter)
			}

			got := adapter.GetAppPath()
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestPackageLevelFunctions(t *testing.T) {
	// Reset singleton for test
	adapterOnce = sync.Once{}
	globalAdapter = nil

	t.Run("GetScenarioPath", func(t *testing.T) {
		path := GetScenarioPath("minimal")
		assert.Contains(t, path, "minimal")
	})

	t.Run("GetModulePath", func(t *testing.T) {
		path := GetModulePath("postgres")
		assert.Contains(t, path, "postgres")
	})

	t.Run("GetOverlayPath", func(t *testing.T) {
		path := GetOverlayPath("production")
		assert.Contains(t, path, "production")
	})

	t.Run("GetScenarioNamespaces", func(t *testing.T) {
		namespaces := GetScenarioNamespaces("minimal")
		assert.NotEmpty(t, namespaces)
		assert.Contains(t, namespaces, "infra")
		assert.Contains(t, namespaces, "app")
	})

	t.Run("ListScenarios", func(t *testing.T) {
		scenarios := ListScenarios()
		assert.NotEmpty(t, scenarios)
		assert.Contains(t, scenarios, "minimal")
	})

	t.Run("ValidateScenario", func(t *testing.T) {
		err := ValidateScenario("minimal")
		assert.NoError(t, err)

		err = ValidateScenario("unknown")
		assert.Error(t, err)
	})
}

func TestAdapter_ConcurrentAccess(t *testing.T) {
	adapter := &Adapter{
		manager: &Manager{
			config: &Config{
				Paths: PathConfig{
					Base:      "/test/base",
					Scenarios: "scenarios",
				},
				Scenarios: map[string]ScenarioConfig{
					"minimal": {
						Name: "minimal",
						Path: "minimal",
					},
				},
			},
		},
	}

	// Test concurrent reads
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = adapter.GetScenarioPath("minimal")
			_ = adapter.ListScenarios()
			_ = adapter.ValidateScenario("minimal")
		}()
	}
	wg.Wait()

	// Should complete without race conditions
	assert.NotNil(t, adapter)
}
