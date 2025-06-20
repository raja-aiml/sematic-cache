package constants

import "path/filepath"

const (
	// K3d specific defaults
	DefaultK3dRegistry = "k3d-local-registry:5000"
	DefaultK3sImage    = "rancher/k3s:v1.28.5-k3s1"

	// Additional namespace constants (core ones are in defaults.go)
	MonitoringNamespace = "monitoring"
	IstioNamespace      = "istio-system"
	LoggingNamespace    = "logging"
	TracingNamespace    = "tracing"

	// Blueprint paths - relative to iaac directory
	BlueprintBasePath      = "blueprint"
	BlueprintInfraPath     = "blueprint/infra"
	BlueprintScenariosPath = "blueprint/scenarios"
	BlueprintModulesPath   = "blueprint/infra/modules"

	// Scenario names
	ScenarioMinimal      = "minimal"
	ScenarioDevelopment  = "development"
	ScenarioServiceMesh  = "service-mesh"
	ScenarioMonitoring   = "monitoring-only"
	ScenarioFullStack    = "full-stack"

	// Component labels
	LabelApp       = "app"
	LabelComponent = "component"
	LabelScenario  = "scenario"
	LabelManaged   = "managed-by"
	ManagedByValue = "iaac"

	// Additional timeouts (core ones are in defaults.go)
	ResourceReadyTimeout    = 600 // seconds
	ClusterCreationTimeout  = 300 // seconds
	DeploymentReadyTimeout  = 600 // seconds
)

// GetBlueprintPath returns the full path to a blueprint component
func GetBlueprintPath(component string) string {
	return filepath.Join(BlueprintBasePath, component)
}

// GetScenarioPath returns the full path to a scenario
func GetScenarioPath(scenario string) string {
	return filepath.Join(BlueprintScenariosPath, scenario)
}

// GetModulePath returns the full path to a module
func GetModulePath(module string) string {
	return filepath.Join(BlueprintModulesPath, module)
}

// GetOverlayPath returns the full path to an overlay
func GetOverlayPath(overlay string) string {
	return filepath.Join(BlueprintInfraPath, "overlays", overlay)
}