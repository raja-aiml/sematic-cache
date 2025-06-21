package validation

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewBlueprintValidator(t *testing.T) {
	tests := []struct {
		name   string
		strict bool
	}{
		{
			name:   "strict_mode",
			strict: true,
		},
		{
			name:   "non_strict_mode",
			strict: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewBlueprintValidator(tt.strict)
			assert.NotNil(t, validator)
			assert.Equal(t, tt.strict, validator.strict)
		})
	}
}

func TestBlueprintValidator_Validate(t *testing.T) {
	tests := []struct {
		name      string
		setup     func() (string, func())
		strict    bool
		checkFunc func(*testing.T, *ValidationResult)
	}{
		{
			name: "blueprint_path_does_not_exist",
			setup: func() (string, func()) {
				return "/non/existent/path", func() {}
			},
			strict: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.False(t, result.IsValid())
				assert.Len(t, result.Errors, 1)
				assert.Contains(t, result.Errors[0], "Blueprint path does not exist")
			},
		},
		{
			name: "valid_blueprint_structure",
			setup: func() (string, func()) {
				tmpDir := t.TempDir()

				// Create required directories
				dirs := []string{
					"scenarios/minimal",
					"scenarios/development",
					"infra",
					"app",
				}
				for _, dir := range dirs {
					err := os.MkdirAll(filepath.Join(tmpDir, dir), 0755)
					require.NoError(t, err)
				}

				// Create kustomization files
				kustomizations := []string{
					"kustomization.yaml",
					"scenarios/minimal/kustomization.yaml",
					"scenarios/development/kustomization.yaml",
				}
				for _, file := range kustomizations {
					content := `apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - base
`
					err := os.WriteFile(filepath.Join(tmpDir, file), []byte(content), 0644)
					require.NoError(t, err)
				}

				return tmpDir, func() {}
			},
			strict: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				// May have errors for missing scenarios
				assert.NotNil(t, result)
				// Should have some info messages
				assert.NotEmpty(t, result.Info)
			},
		},
		{
			name: "missing_required_directories",
			setup: func() (string, func()) {
				tmpDir := t.TempDir()
				// Only create scenarios directory
				err := os.MkdirAll(filepath.Join(tmpDir, "scenarios"), 0755)
				require.NoError(t, err)
				return tmpDir, func() {}
			},
			strict: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.False(t, result.IsValid())
				assert.GreaterOrEqual(t, len(result.Errors), 2)
				// Should have errors for missing infra and app directories
				errorsStr := result.Errors[0] + result.Errors[1]
				assert.Contains(t, errorsStr, "infra")
				assert.Contains(t, errorsStr, "app")
			},
		},
		{
			name: "strict_mode_missing_optional",
			setup: func() (string, func()) {
				tmpDir := t.TempDir()
				// Create only required directories
				dirs := []string{"scenarios", "infra", "app"}
				for _, dir := range dirs {
					err := os.MkdirAll(filepath.Join(tmpDir, dir), 0755)
					require.NoError(t, err)
				}
				return tmpDir, func() {}
			},
			strict: true,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				// Should have warnings for missing optional directories in strict mode
				assert.NotEmpty(t, result.Warnings)
			},
		},
		{
			name: "invalid_kustomization_yaml",
			setup: func() (string, func()) {
				tmpDir := t.TempDir()

				// Create basic structure
				dirs := []string{"scenarios/minimal", "infra", "app"}
				for _, dir := range dirs {
					err := os.MkdirAll(filepath.Join(tmpDir, dir), 0755)
					require.NoError(t, err)
				}

				// Create invalid kustomization.yaml
				err := os.WriteFile(
					filepath.Join(tmpDir, "kustomization.yaml"),
					[]byte("invalid: yaml: content:"),
					0644,
				)
				require.NoError(t, err)

				return tmpDir, func() {}
			},
			strict: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.False(t, result.IsValid())
				// Should have error about invalid kustomization
				found := false
				for _, err := range result.Errors {
					if contains(err, "kustomization") && contains(err, "Invalid") {
						found = true
						break
					}
				}
				assert.True(t, found, "Should have error about invalid kustomization")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path, cleanup := tt.setup()
			defer cleanup()

			validator := NewBlueprintValidator(tt.strict)
			result, err := validator.Validate(path)

			require.NoError(t, err)
			require.NotNil(t, result)

			if tt.checkFunc != nil {
				tt.checkFunc(t, result)
			}
		})
	}
}

func TestBlueprintValidator_validateKustomizationFile(t *testing.T) {
	tests := []struct {
		name    string
		content string
		wantErr bool
	}{
		{
			name: "valid_kustomization",
			content: `apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml
`,
			wantErr: false,
		},
		{
			name: "missing_apiVersion",
			content: `kind: Kustomization
resources:
  - deployment.yaml
`,
			wantErr: true,
		},
		{
			name: "missing_kind",
			content: `apiVersion: kustomize.config.k8s.io/v1beta1
resources:
  - deployment.yaml
`,
			wantErr: true,
		},
		{
			name: "wrong_kind",
			content: `apiVersion: kustomize.config.k8s.io/v1beta1
kind: NotKustomization
resources:
  - deployment.yaml
`,
			wantErr: true,
		},
		{
			name:    "invalid_yaml",
			content: `invalid: yaml: content:`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpFile := filepath.Join(t.TempDir(), "kustomization.yaml")
			err := os.WriteFile(tmpFile, []byte(tt.content), 0644)
			require.NoError(t, err)

			validator := NewBlueprintValidator(false)
			result := NewValidationResult()
			err = validator.validateKustomizationFile(tmpFile, result)

			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestBlueprintValidator_validateScenarios(t *testing.T) {
	validator := NewBlueprintValidator(false)

	tmpDir := t.TempDir()
	scenariosDir := filepath.Join(tmpDir, "scenarios")

	// Create some scenarios
	scenarios := []string{"minimal", "development"}
	for _, scenario := range scenarios {
		scenarioPath := filepath.Join(scenariosDir, scenario)
		err := os.MkdirAll(scenarioPath, 0755)
		require.NoError(t, err)

		// Create kustomization.yaml
		kustomContent := `apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../infra/base
  - ../../app/base
`
		err = os.WriteFile(
			filepath.Join(scenarioPath, "kustomization.yaml"),
			[]byte(kustomContent),
			0644,
		)
		require.NoError(t, err)
	}

	result := NewValidationResult()
	validator.validateScenarios(tmpDir, result)

	// Should have errors for missing scenarios
	assert.NotEmpty(t, result.Errors)
	for _, err := range result.Errors {
		// All errors should be about missing scenarios
		assert.Contains(t, err, "Missing scenario")
	}
}

func TestBlueprintValidator_checkScenarioReferences(t *testing.T) {
	tests := []struct {
		name          string
		scenario      string
		kustomization string
		expectedRefs  []string
		strict        bool
		expectWarning bool
	}{
		{
			name:     "minimal_scenario_correct_refs",
			scenario: "minimal",
			kustomization: `apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../infra/base
  - ../../app/base
`,
			expectedRefs:  []string{"infra/base", "app/base"},
			strict:        true,
			expectWarning: false,
		},
		{
			name:     "development_scenario_missing_ref",
			scenario: "development",
			kustomization: `apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../infra/base
`,
			expectedRefs:  []string{"modules/dev-tools"},
			strict:        true,
			expectWarning: true,
		},
		{
			name:     "full_stack_scenario_with_all_refs",
			scenario: "full-stack",
			kustomization: `apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../infra/modules/istio
  - ../../infra/modules/observability
  - ../../infra/modules/dev-tools
`,
			expectedRefs:  []string{"modules/istio", "modules/observability", "modules/dev-tools"},
			strict:        true,
			expectWarning: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			scenarioPath := filepath.Join(tmpDir, tt.scenario)
			err := os.MkdirAll(scenarioPath, 0755)
			require.NoError(t, err)

			err = os.WriteFile(
				filepath.Join(scenarioPath, "kustomization.yaml"),
				[]byte(tt.kustomization),
				0644,
			)
			require.NoError(t, err)

			validator := NewBlueprintValidator(tt.strict)
			result := NewValidationResult()

			validator.checkScenarioReferences(scenarioPath, tt.expectedRefs, result)

			if tt.expectWarning {
				assert.NotEmpty(t, result.Warnings)
			} else {
				assert.Empty(t, result.Warnings)
			}
		})
	}
}

func TestBlueprintValidator_validateModules(t *testing.T) {
	tests := []struct {
		name           string
		strict         bool
		createModules  []string
		expectWarnings bool
	}{
		{
			name:           "all_modules_present",
			strict:         true,
			createModules:  []string{"observability", "istio", "dev-tools", "security"},
			expectWarnings: false,
		},
		{
			name:           "missing_modules_strict",
			strict:         true,
			createModules:  []string{"observability"},
			expectWarnings: true,
		},
		{
			name:           "missing_modules_non_strict",
			strict:         false,
			createModules:  []string{"observability"},
			expectWarnings: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			modulesPath := filepath.Join(tmpDir, "infra", "modules")

			for _, module := range tt.createModules {
				modulePath := filepath.Join(modulesPath, module)
				err := os.MkdirAll(modulePath, 0755)
				require.NoError(t, err)

				// Create kustomization.yaml
				kustomContent := `apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
`
				err = os.WriteFile(
					filepath.Join(modulePath, "kustomization.yaml"),
					[]byte(kustomContent),
					0644,
				)
				require.NoError(t, err)
			}

			validator := NewBlueprintValidator(tt.strict)
			result := NewValidationResult()
			validator.validateModules(tmpDir, result)

			if tt.expectWarnings {
				assert.NotEmpty(t, result.Warnings)
			} else {
				assert.Empty(t, result.Warnings)
			}
		})
	}
}

func TestBlueprintValidator_validateOverlays(t *testing.T) {
	tests := []struct {
		name           string
		strict         bool
		createOverlays []string
		expectWarnings bool
	}{
		{
			name:           "all_overlays_present",
			strict:         true,
			createOverlays: []string{"local", "dev"},
			expectWarnings: false,
		},
		{
			name:           "missing_overlays_strict",
			strict:         true,
			createOverlays: []string{"local"},
			expectWarnings: true,
		},
		{
			name:           "missing_overlays_non_strict",
			strict:         false,
			createOverlays: []string{"local"},
			expectWarnings: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			overlaysPath := filepath.Join(tmpDir, "infra", "overlays")

			for _, overlay := range tt.createOverlays {
				overlayPath := filepath.Join(overlaysPath, overlay)
				err := os.MkdirAll(overlayPath, 0755)
				require.NoError(t, err)

				// Create kustomization.yaml
				kustomContent := `apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
`
				err = os.WriteFile(
					filepath.Join(overlayPath, "kustomization.yaml"),
					[]byte(kustomContent),
					0644,
				)
				require.NoError(t, err)
			}

			validator := NewBlueprintValidator(tt.strict)
			result := NewValidationResult()
			validator.validateOverlays(tmpDir, result)

			if tt.expectWarnings {
				assert.NotEmpty(t, result.Warnings)
			} else {
				assert.Empty(t, result.Warnings)
			}
		})
	}
}

// Helper function
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
