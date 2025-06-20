package constants

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestScenarioConstants(t *testing.T) {
	t.Run("all scenarios defined", func(t *testing.T) {
		assert.Equal(t, "minimal", ScenarioMinimal)
		assert.Equal(t, "development", ScenarioDevelopment)
		assert.Equal(t, "service-mesh", ScenarioServiceMesh)
		assert.Equal(t, "monitoring-only", ScenarioMonitoring)
		assert.Equal(t, "full-stack", ScenarioFullStack)
	})
}

func TestNamespaceConstants(t *testing.T) {
	t.Run("all namespaces defined", func(t *testing.T) {
		// Core namespaces (from defaults.go)
		assert.NotEmpty(t, InfraNamespace)
		assert.NotEmpty(t, AppNamespace)
		
		// Additional namespaces
		assert.Equal(t, "monitoring", MonitoringNamespace)
		assert.Equal(t, "istio-system", IstioNamespace)
		assert.Equal(t, "logging", LoggingNamespace)
		assert.Equal(t, "tracing", TracingNamespace)
	})
}

func TestBlueprintPaths(t *testing.T) {
	t.Run("path constants", func(t *testing.T) {
		assert.Equal(t, "blueprint", BlueprintBasePath)
		assert.Equal(t, "blueprint/infra", BlueprintInfraPath)
		assert.Equal(t, "blueprint/scenarios", BlueprintScenariosPath)
		assert.Equal(t, "blueprint/infra/modules", BlueprintModulesPath)
	})
}

func TestGetBlueprintPath(t *testing.T) {
	tests := []struct {
		name      string
		component string
		expected  string
	}{
		{
			name:      "simple component",
			component: "base",
			expected:  filepath.Join("blueprint", "base"),
		},
		{
			name:      "nested component",
			component: "infra/overlays",
			expected:  filepath.Join("blueprint", "infra", "overlays"),
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetBlueprintPath(tt.component)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestGetScenarioPath(t *testing.T) {
	scenarios := []string{
		ScenarioMinimal,
		ScenarioDevelopment,
		ScenarioServiceMesh,
		ScenarioMonitoring,
		ScenarioFullStack,
	}
	
	for _, scenario := range scenarios {
		t.Run("scenario_"+scenario, func(t *testing.T) {
			path := GetScenarioPath(scenario)
			assert.True(t, strings.HasPrefix(path, BlueprintScenariosPath))
			assert.True(t, strings.HasSuffix(path, scenario))
			assert.Equal(t, filepath.Join(BlueprintScenariosPath, scenario), path)
		})
	}
}

func TestGetModulePath(t *testing.T) {
	modules := []string{
		"observability",
		"istio",
		"security",
		"networking",
	}
	
	for _, module := range modules {
		t.Run("module_"+module, func(t *testing.T) {
			path := GetModulePath(module)
			assert.True(t, strings.HasPrefix(path, BlueprintModulesPath))
			assert.True(t, strings.HasSuffix(path, module))
			assert.Equal(t, filepath.Join(BlueprintModulesPath, module), path)
		})
	}
}

func TestGetOverlayPath(t *testing.T) {
	overlays := []string{
		"local",
		"dev",
		"staging",
		"production",
	}
	
	for _, overlay := range overlays {
		t.Run("overlay_"+overlay, func(t *testing.T) {
			path := GetOverlayPath(overlay)
			assert.True(t, strings.HasPrefix(path, BlueprintInfraPath))
			assert.True(t, strings.Contains(path, "overlays"))
			assert.True(t, strings.HasSuffix(path, overlay))
			assert.Equal(t, filepath.Join(BlueprintInfraPath, "overlays", overlay), path)
		})
	}
}

func TestTimeoutConstants(t *testing.T) {
	t.Run("timeout values are reasonable", func(t *testing.T) {
		assert.Greater(t, ResourceReadyTimeout, 0)
		assert.Greater(t, ClusterCreationTimeout, 0)
		assert.Greater(t, DeploymentReadyTimeout, 0)
		
		// Deployment timeout should be longer than cluster creation
		assert.GreaterOrEqual(t, DeploymentReadyTimeout, ClusterCreationTimeout)
	})
}

func TestLabelConstants(t *testing.T) {
	t.Run("label constants defined", func(t *testing.T) {
		assert.Equal(t, "app", LabelApp)
		assert.Equal(t, "component", LabelComponent)
		assert.Equal(t, "scenario", LabelScenario)
		assert.Equal(t, "managed-by", LabelManaged)
		assert.Equal(t, "iaac", ManagedByValue)
	})
}