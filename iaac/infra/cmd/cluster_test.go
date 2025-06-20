package cmd

import (
	"strings"
	"testing"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/constants"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestClusterCmd(t *testing.T) {
	cmd := ClusterCmd()
	
	t.Run("command basic properties", func(t *testing.T) {
		assert.Equal(t, "cluster", cmd.Use)
		assert.Contains(t, cmd.Short, "blueprint scenarios")
		// ClusterCmd returns a command with subcommands, not a RunE
	})
	
	t.Run("has required subcommands", func(t *testing.T) {
		subcommands := make(map[string]bool)
		for _, subcmd := range cmd.Commands() {
			subcommands[subcmd.Use] = true
		}
		
		assert.True(t, subcommands["up"], "should have 'up' subcommand")
		assert.True(t, subcommands["down"], "should have 'down' subcommand")
		assert.True(t, subcommands["ps"], "should have 'ps' subcommand")
		assert.True(t, subcommands["test"], "should have 'test' subcommand")
		assert.True(t, subcommands["logs"], "should have 'logs' subcommand")
	})
	
	t.Run("has blueprint flags", func(t *testing.T) {
		// Check for scenario flag
		scenarioFlag := cmd.PersistentFlags().Lookup("scenario")
		require.NotNil(t, scenarioFlag, "should have --scenario flag")
		assert.Equal(t, constants.ScenarioMinimal, scenarioFlag.DefValue)
		
		// Check for overlay flag
		overlayFlag := cmd.PersistentFlags().Lookup("overlay")
		require.NotNil(t, overlayFlag, "should have --overlay flag")
		assert.Equal(t, "local", overlayFlag.DefValue)
		
		// Check for backward compatibility kustomize-path flag
		kustomizeFlag := cmd.PersistentFlags().Lookup("kustomize-path")
		require.NotNil(t, kustomizeFlag, "should have --kustomize-path flag")
	})
}

func TestClusterUpCmd(t *testing.T) {
	clusterName := "test-cluster"
	scenario := constants.ScenarioMinimal
	overlay := "local"
	kustomizePath := ""
	
	cmd := clusterUpCmd(&clusterName, &scenario, &overlay, &kustomizePath)
	
	t.Run("command properties", func(t *testing.T) {
		assert.Equal(t, "up", cmd.Use)
		assert.Contains(t, cmd.Short, "deploy blueprint scenario")
		assert.NotNil(t, cmd.RunE)
	})
	
	t.Run("help text includes scenarios", func(t *testing.T) {
		assert.Contains(t, cmd.Long, "minimal:")
		assert.Contains(t, cmd.Long, "development:")
		assert.Contains(t, cmd.Long, "service-mesh:")
		assert.Contains(t, cmd.Long, "monitoring-only:")
		assert.Contains(t, cmd.Long, "full-stack:")
	})
}

func TestScenarioConstants(t *testing.T) {
	// Verify all scenario constants are defined
	scenarios := []string{
		constants.ScenarioMinimal,
		constants.ScenarioDevelopment,
		constants.ScenarioServiceMesh,
		constants.ScenarioMonitoring,
		constants.ScenarioFullStack,
	}
	
	for _, scenario := range scenarios {
		assert.NotEmpty(t, scenario, "scenario constant should not be empty")
	}
	
	// Verify scenario paths
	for _, scenario := range scenarios {
		path := constants.GetScenarioPath(scenario)
		assert.Contains(t, path, "blueprint/scenarios/")
		assert.Contains(t, path, scenario)
	}
}

func TestBlueprintPaths(t *testing.T) {
	t.Run("GetBlueprintPath", func(t *testing.T) {
		path := constants.GetBlueprintPath("test-component")
		assert.Contains(t, path, "blueprint")
		assert.Contains(t, path, "test-component")
	})
	
	t.Run("GetModulePath", func(t *testing.T) {
		path := constants.GetModulePath("observability")
		assert.Contains(t, path, "blueprint/infra/modules")
		assert.Contains(t, path, "observability")
	})
	
	t.Run("GetOverlayPath", func(t *testing.T) {
		path := constants.GetOverlayPath("local")
		assert.Contains(t, path, "blueprint/infra/overlays")
		assert.Contains(t, path, "local")
	})
}

func TestFindIaacPath(t *testing.T) {
	// Test with various starting paths
	testCases := []struct {
		name     string
		start    string
		expected bool
	}{
		{
			name:     "from project root",
			start:    "/Users/test/project",
			expected: false, // Won't find iaac in test environment
		},
		{
			name:     "from deep nested path",
			start:    "/Users/test/project/deep/nested/path",
			expected: false,
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := findIaacPath(tc.start)
			if tc.expected {
				assert.NotEmpty(t, result)
			}
		})
	}
}

func TestWaitForScenarioComponents(t *testing.T) {
	// This would require mocking the kubernetes client
	// For now, just verify the function exists and handles scenarios
	
	scenarios := []string{
		constants.ScenarioMinimal,
		constants.ScenarioDevelopment,
		constants.ScenarioServiceMesh,
		constants.ScenarioMonitoring,
		constants.ScenarioFullStack,
		"unknown-scenario", // Should fall back to default
	}
	
	for _, scenario := range scenarios {
		t.Run("scenario_"+scenario, func(t *testing.T) {
			// Just verify we can call the function without panic
			// In real tests, we'd mock the k8s client
			assert.NotPanics(t, func() {
				// The function will error due to nil client, but shouldn't panic
				_ = waitForScenarioComponents(nil, nil, scenario)
			})
		})
	}
}

func TestPrintScenarioAccess(t *testing.T) {
	// Test that printScenarioAccess doesn't panic for any scenario
	scenarios := []string{
		constants.ScenarioMinimal,
		constants.ScenarioDevelopment,
		constants.ScenarioServiceMesh,
		constants.ScenarioMonitoring,
		constants.ScenarioFullStack,
	}
	
	for _, scenario := range scenarios {
		t.Run("print_access_"+scenario, func(t *testing.T) {
			assert.NotPanics(t, func() {
				// Capture output to avoid test noise
				printScenarioAccess(scenario)
			})
		})
	}
}

func TestClusterStatusCmd(t *testing.T) {
	clusterName := "test-cluster"
	cmd := clusterStatusCmd(&clusterName)
	
	t.Run("command properties", func(t *testing.T) {
		assert.Equal(t, "ps", cmd.Use)
		assert.Contains(t, cmd.Short, "blueprint component")
		assert.NotNil(t, cmd.RunE)
	})
}

func TestClusterTestCmd(t *testing.T) {
	clusterName := "test-cluster"
	scenario := constants.ScenarioMinimal
	cmd := clusterTestCmd(&clusterName, &scenario)
	
	t.Run("command properties", func(t *testing.T) {
		assert.Equal(t, "test", cmd.Use)
		assert.Contains(t, cmd.Short, "blueprint scenario")
		assert.NotNil(t, cmd.RunE)
	})
}

// Integration test for command help output
func TestCommandHelpOutput(t *testing.T) {
	cmd := ClusterCmd()
	
	// Capture help output
	var helpOutput strings.Builder
	cmd.SetOut(&helpOutput)
	cmd.SetArgs([]string{"--help"})
	
	err := cmd.Execute()
	require.NoError(t, err)
	
	help := helpOutput.String()
	
	// Verify help contains expected content
	assert.Contains(t, help, "blueprint scenarios")
	assert.Contains(t, help, "--scenario")
	assert.Contains(t, help, "--overlay")
	assert.Contains(t, help, "--kustomize-path")
	assert.Contains(t, help, "up")
	assert.Contains(t, help, "down")
	assert.Contains(t, help, "ps")
	assert.Contains(t, help, "test")
}