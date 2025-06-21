package validation

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/blueprint"
	"gopkg.in/yaml.v3"
)

// BlueprintValidator validates blueprint directory structure
type BlueprintValidator struct {
	strict bool
}

// NewBlueprintValidator creates a new blueprint validator
func NewBlueprintValidator(strict bool) *BlueprintValidator {
	return &BlueprintValidator{
		strict: strict,
	}
}

// Validate validates a blueprint directory
func (v *BlueprintValidator) Validate(blueprintPath string) (*ValidationResult, error) {
	result := NewValidationResult()

	// Check if path exists
	if _, err := os.Stat(blueprintPath); os.IsNotExist(err) {
		result.AddError("Blueprint path does not exist: %s", blueprintPath)
		return result, nil
	}

	// Try to load blueprint configuration
	var manager *blueprint.Manager
	configPath, err := blueprint.FindBlueprintConfig(blueprintPath)
	if err == nil {
		manager, err = blueprint.NewManager(configPath)
		if err != nil {
			result.AddWarning("Found blueprint config but failed to load it: %v", err)
		} else {
			result.AddInfo("Using blueprint configuration from: %s", configPath)
			// Use config-driven validation
			return v.validateWithConfig(manager, result)
		}
	}

	// Fallback to hardcoded validation
	result.AddInfo("No blueprint config found, using default validation")
	v.validateStructure(blueprintPath, result)
	v.validateKustomizations(blueprintPath, result)
	v.validateScenarios(blueprintPath, result)
	v.validateModules(blueprintPath, result)
	v.validateOverlays(blueprintPath, result)

	return result, nil
}

// validateWithConfig validates using blueprint configuration
func (v *BlueprintValidator) validateWithConfig(manager *blueprint.Manager, result *ValidationResult) (*ValidationResult, error) {
	config := manager.GetConfig()

	// Validate structure based on config
	if err := manager.ValidateStructure(); err != nil {
		result.AddError("Structure validation failed: %v", err)
	}

	// Validate scenarios exist
	for name, scenario := range config.Scenarios {
		scenarioPath := config.GetScenarioPath(name)
		if _, err := os.Stat(scenarioPath); os.IsNotExist(err) {
			result.AddError("Scenario path does not exist: %s -> %s", name, scenarioPath)
		} else {
			result.AddInfo("Found scenario: %s (%s)", name, scenario.Description)
			// Check for kustomization.yaml
			kustomizePath := filepath.Join(scenarioPath, "kustomization.yaml")
			if _, err := os.Stat(kustomizePath); os.IsNotExist(err) {
				result.AddError("Scenario %s missing kustomization.yaml", name)
			}
		}
	}

	// Validate modules exist
	for name, module := range config.Modules {
		modulePath := config.GetModulePath(name)
		if _, err := os.Stat(modulePath); os.IsNotExist(err) {
			if v.strict {
				result.AddWarning("Module path does not exist: %s -> %s", name, modulePath)
			}
		} else {
			result.AddInfo("Found module: %s (%s)", name, module.Description)
		}
	}

	// Validate overlays exist
	for name, overlay := range config.Overlays {
		overlayPath := config.GetOverlayPath(name)
		if _, err := os.Stat(overlayPath); os.IsNotExist(err) {
			if v.strict {
				result.AddWarning("Overlay path does not exist: %s -> %s", name, overlayPath)
			}
		} else {
			result.AddInfo("Found overlay: %s (%s)", name, overlay.Description)
		}
	}

	return result, nil
}

// validateStructure checks the required directory structure
func (v *BlueprintValidator) validateStructure(blueprintPath string, result *ValidationResult) {
	requiredDirs := []string{
		"scenarios",
		"infra",
		"app",
	}

	optionalDirs := []string{
		"infra/modules",
		"infra/overlays",
		"validation-kit",
		"hack",
	}

	// Check required directories
	for _, dir := range requiredDirs {
		path := filepath.Join(blueprintPath, dir)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			result.AddError("Required directory missing: %s", dir)
		} else {
			result.AddInfo("Found required directory: %s", dir)
		}
	}

	// Check optional directories
	for _, dir := range optionalDirs {
		path := filepath.Join(blueprintPath, dir)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			if v.strict {
				result.AddWarning("Optional directory missing: %s", dir)
			}
		} else {
			result.AddInfo("Found optional directory: %s", dir)
		}
	}
}

// validateKustomizations checks for valid kustomization.yaml files
func (v *BlueprintValidator) validateKustomizations(blueprintPath string, result *ValidationResult) {
	// Paths that should have kustomization.yaml
	kustomizePaths := []string{
		".",
		"scenarios/minimal",
		"scenarios/development",
		"scenarios/service-mesh",
		"scenarios/monitoring-only",
		"scenarios/full-stack",
		"infra",
		"app",
		"infra/base",
		"infra/overlays/local",
		"infra/overlays/dev",
	}

	for _, path := range kustomizePaths {
		fullPath := filepath.Join(blueprintPath, path, "kustomization.yaml")
		if err := v.validateKustomizationFile(fullPath, result); err != nil {
			// Only error for required paths
			if strings.HasPrefix(path, "scenarios/") || path == "." {
				result.AddError("Invalid or missing kustomization.yaml in %s: %v", path, err)
			} else if v.strict {
				result.AddWarning("Invalid or missing kustomization.yaml in %s: %v", path, err)
			}
		}
	}
}

// validateKustomizationFile validates a single kustomization.yaml file
func (v *BlueprintValidator) validateKustomizationFile(path string, result *ValidationResult) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var kustomization map[string]interface{}
	if err := yaml.Unmarshal(data, &kustomization); err != nil {
		return fmt.Errorf("invalid YAML: %w", err)
	}

	// Check for required fields
	if _, ok := kustomization["apiVersion"]; !ok {
		return fmt.Errorf("missing apiVersion field")
	}

	if _, ok := kustomization["kind"]; !ok {
		return fmt.Errorf("missing kind field")
	}

	// Validate kind
	if kind, ok := kustomization["kind"].(string); ok {
		if kind != "Kustomization" {
			return fmt.Errorf("invalid kind: %s (expected Kustomization)", kind)
		}
	}

	return nil
}

// validateScenarios validates scenario definitions
func (v *BlueprintValidator) validateScenarios(blueprintPath string, result *ValidationResult) {
	scenariosPath := filepath.Join(blueprintPath, "scenarios")

	expectedScenarios := []string{
		"minimal",
		"development",
		"service-mesh",
		"monitoring-only",
		"full-stack",
	}

	for _, scenario := range expectedScenarios {
		scenarioPath := filepath.Join(scenariosPath, scenario)
		if _, err := os.Stat(scenarioPath); os.IsNotExist(err) {
			result.AddError("Missing scenario: %s", scenario)
			continue
		}

		// Check for kustomization.yaml
		kustomizePath := filepath.Join(scenarioPath, "kustomization.yaml")
		if _, err := os.Stat(kustomizePath); os.IsNotExist(err) {
			result.AddError("Scenario %s missing kustomization.yaml", scenario)
		}

		// Validate scenario-specific requirements
		v.validateScenarioRequirements(scenario, scenarioPath, result)
	}
}

// validateScenarioRequirements checks scenario-specific requirements
func (v *BlueprintValidator) validateScenarioRequirements(scenario, path string, result *ValidationResult) {
	switch scenario {
	case "minimal":
		// Minimal should only reference base components
		v.checkScenarioReferences(path, []string{"infra/base", "app/base"}, result)

	case "development":
		// Development should include dev tools
		v.checkScenarioReferences(path, []string{"modules/dev-tools"}, result)

	case "service-mesh":
		// Service mesh should include Istio
		v.checkScenarioReferences(path, []string{"modules/istio"}, result)

	case "monitoring-only":
		// Monitoring should include observability components
		v.checkScenarioReferences(path, []string{"modules/observability/monitoring"}, result)

	case "full-stack":
		// Full stack should include everything
		v.checkScenarioReferences(path, []string{
			"modules/istio",
			"modules/observability",
			"modules/dev-tools",
		}, result)
	}
}

// checkScenarioReferences verifies that a scenario references expected components
func (v *BlueprintValidator) checkScenarioReferences(scenarioPath string, expectedRefs []string, result *ValidationResult) {
	kustomizePath := filepath.Join(scenarioPath, "kustomization.yaml")
	data, err := os.ReadFile(kustomizePath)
	if err != nil {
		return
	}

	var kustomization map[string]interface{}
	if err := yaml.Unmarshal(data, &kustomization); err != nil {
		return
	}

	// Get resources/bases
	var references []string
	if resources, ok := kustomization["resources"].([]interface{}); ok {
		for _, r := range resources {
			if str, ok := r.(string); ok {
				references = append(references, str)
			}
		}
	}
	if bases, ok := kustomization["bases"].([]interface{}); ok {
		for _, b := range bases {
			if str, ok := b.(string); ok {
				references = append(references, str)
			}
		}
	}

	// Check for expected references
	for _, expected := range expectedRefs {
		found := false
		for _, ref := range references {
			if strings.Contains(ref, expected) {
				found = true
				break
			}
		}
		if !found && v.strict {
			result.AddWarning("Scenario %s missing expected reference: %s",
				filepath.Base(scenarioPath), expected)
		}
	}
}

// validateModules checks module definitions
func (v *BlueprintValidator) validateModules(blueprintPath string, result *ValidationResult) {
	modulesPath := filepath.Join(blueprintPath, "infra", "modules")

	if _, err := os.Stat(modulesPath); os.IsNotExist(err) {
		if v.strict {
			result.AddWarning("Modules directory not found")
		}
		return
	}

	// Check for expected modules
	expectedModules := []string{
		"observability",
		"istio",
		"dev-tools",
		"security",
	}

	for _, module := range expectedModules {
		modulePath := filepath.Join(modulesPath, module)
		if _, err := os.Stat(modulePath); os.IsNotExist(err) {
			if v.strict {
				result.AddWarning("Expected module not found: %s", module)
			}
		} else {
			// Validate module has kustomization.yaml
			kustomizePath := filepath.Join(modulePath, "kustomization.yaml")
			if _, err := os.Stat(kustomizePath); os.IsNotExist(err) {
				result.AddError("Module %s missing kustomization.yaml", module)
			}
		}
	}
}

// validateOverlays checks overlay definitions
func (v *BlueprintValidator) validateOverlays(blueprintPath string, result *ValidationResult) {
	overlaysPath := filepath.Join(blueprintPath, "infra", "overlays")

	if _, err := os.Stat(overlaysPath); os.IsNotExist(err) {
		if v.strict {
			result.AddWarning("Overlays directory not found")
		}
		return
	}

	// Check for expected overlays
	expectedOverlays := []string{"local", "dev"}

	for _, overlay := range expectedOverlays {
		overlayPath := filepath.Join(overlaysPath, overlay)
		if _, err := os.Stat(overlayPath); os.IsNotExist(err) {
			if v.strict {
				result.AddWarning("Expected overlay not found: %s", overlay)
			}
		} else {
			// Validate overlay has kustomization.yaml
			kustomizePath := filepath.Join(overlayPath, "kustomization.yaml")
			if _, err := os.Stat(kustomizePath); os.IsNotExist(err) {
				result.AddError("Overlay %s missing kustomization.yaml", overlay)
			}
		}
	}
}
