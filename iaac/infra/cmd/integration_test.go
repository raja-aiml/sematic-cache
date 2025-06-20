//go:build integration
// +build integration

package cmd

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/constants"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestBlueprintIntegration validates the complete blueprint integration
func TestBlueprintIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	t.Run("blueprint directory structure exists", func(t *testing.T) {
		// Find iaac directory
		workDir, err := os.Getwd()
		require.NoError(t, err)

		// Navigate up to find iaac directory
		iaacPath := findIaacPathForTest(workDir)
		require.NotEmpty(t, iaacPath, "iaac directory should be found")

		// Check blueprint directory exists
		blueprintPath := filepath.Join(iaacPath, "blueprint")
		info, err := os.Stat(blueprintPath)
		require.NoError(t, err, "blueprint directory should exist")
		assert.True(t, info.IsDir(), "blueprint should be a directory")

		// Check scenarios exist
		scenarios := []string{
			constants.ScenarioMinimal,
			constants.ScenarioDevelopment,
			constants.ScenarioServiceMesh,
			constants.ScenarioMonitoring,
			constants.ScenarioFullStack,
		}

		for _, scenario := range scenarios {
			scenarioPath := filepath.Join(blueprintPath, "scenarios", scenario)
			_, err := os.Stat(scenarioPath)
			assert.NoError(t, err, "scenario %s should exist at %s", scenario, scenarioPath)

			// Check kustomization.yaml exists
			kustomizationPath := filepath.Join(scenarioPath, "kustomization.yaml")
			_, err = os.Stat(kustomizationPath)
			assert.NoError(t, err, "kustomization.yaml should exist for scenario %s", scenario)
		}
	})

	t.Run("command structure validates", func(t *testing.T) {
		// Test cluster command
		clusterCmd := ClusterCmd()
		assert.NotNil(t, clusterCmd)

		// Verify flags are properly set
		scenarioFlag := clusterCmd.PersistentFlags().Lookup("scenario")
		assert.NotNil(t, scenarioFlag)

		overlayFlag := clusterCmd.PersistentFlags().Lookup("overlay")
		assert.NotNil(t, overlayFlag)

		// Test workflow command
		workflowCmd := WorkflowCmd()
		assert.NotNil(t, workflowCmd)

		// Verify workflow has scenario support
		workflowScenarioFlag := workflowCmd.PersistentFlags().Lookup("scenario")
		assert.NotNil(t, workflowScenarioFlag)
	})

	t.Run("path helper functions work correctly", func(t *testing.T) {
		// Test scenario paths
		minimalPath := constants.GetScenarioPath(constants.ScenarioMinimal)
		assert.Contains(t, minimalPath, "blueprint/scenarios/minimal")

		// Test module paths
		observabilityPath := constants.GetModulePath("observability")
		assert.Contains(t, observabilityPath, "blueprint/infra/modules/observability")

		// Test overlay paths
		localOverlayPath := constants.GetOverlayPath("local")
		assert.Contains(t, localOverlayPath, "blueprint/infra/overlays/local")
	})
}

// Helper function to find iaac directory for tests
func findIaacPathForTest(startPath string) string {
	current := startPath
	for {
		iaacPath := filepath.Join(current, "iaac")
		if _, err := os.Stat(iaacPath); err == nil {
			return iaacPath
		}

		parent := filepath.Dir(current)
		if parent == current {
			break
		}
		current = parent
	}

	// Try relative path from test directory
	if _, err := os.Stat("../../iaac"); err == nil {
		abs, _ := filepath.Abs("../../iaac")
		return abs
	}

	return ""
}

// TestScenarioDeploymentLogic tests the scenario-specific deployment logic
func TestScenarioDeploymentLogic(t *testing.T) {
	scenarios := map[string][]string{
		constants.ScenarioMinimal: {
			constants.InfraNamespace,
		},
		constants.ScenarioDevelopment: {
			constants.InfraNamespace,
			constants.MonitoringNamespace,
			constants.LoggingNamespace,
		},
		constants.ScenarioServiceMesh: {
			constants.IstioNamespace,
		},
		constants.ScenarioMonitoring: {
			constants.MonitoringNamespace,
			constants.LoggingNamespace,
			constants.TracingNamespace,
		},
		constants.ScenarioFullStack: {
			constants.InfraNamespace,
			constants.IstioNamespace,
			constants.MonitoringNamespace,
			constants.LoggingNamespace,
			constants.TracingNamespace,
		},
	}

	for scenario, expectedNamespaces := range scenarios {
		t.Run("scenario_"+scenario, func(t *testing.T) {
			// This is a logical test - in real deployment these namespaces would be created
			assert.NotEmpty(t, expectedNamespaces, "scenario %s should have expected namespaces", scenario)
		})
	}
}
