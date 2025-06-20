package cmd

import (
	"testing"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/constants"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWorkflowCmd(t *testing.T) {
	cmd := WorkflowCmd()
	
	t.Run("command basic properties", func(t *testing.T) {
		assert.Equal(t, "workflow", cmd.Use)
		assert.Contains(t, cmd.Short, "workflow orchestrator")
		// WorkflowCmd returns a command with subcommands, not a RunE
	})
	
	t.Run("has required subcommands", func(t *testing.T) {
		subcommands := make(map[string]bool)
		for _, subcmd := range cmd.Commands() {
			subcommands[subcmd.Use] = true
		}
		
		assert.True(t, subcommands["full"], "should have 'full' subcommand")
		assert.True(t, subcommands["setup"], "should have 'setup' subcommand")
		assert.True(t, subcommands["build"], "should have 'build' subcommand")
		assert.True(t, subcommands["deploy"], "should have 'deploy' subcommand")
		assert.True(t, subcommands["test"], "should have 'test' subcommand")
		assert.True(t, subcommands["cleanup"], "should have 'cleanup' subcommand")
		assert.True(t, subcommands["status"], "should have 'status' subcommand")
		assert.True(t, subcommands["logs"], "should have 'logs' subcommand")
		assert.True(t, subcommands["reset"], "should have 'reset' subcommand")
	})
	
	t.Run("has blueprint integration flags", func(t *testing.T) {
		// Check for scenario flag
		scenarioFlag := cmd.PersistentFlags().Lookup("scenario")
		require.NotNil(t, scenarioFlag, "should have --scenario flag")
		assert.Equal(t, constants.ScenarioDevelopment, scenarioFlag.DefValue, "workflow should default to development scenario")
		
		// Check for overlay flag
		overlayFlag := cmd.PersistentFlags().Lookup("overlay")
		require.NotNil(t, overlayFlag, "should have --overlay flag")
		assert.Equal(t, "local", overlayFlag.DefValue)
		
		// Check existing flags
		clusterFlag := cmd.PersistentFlags().Lookup("cluster")
		require.NotNil(t, clusterFlag, "should have --cluster flag")
		
		imageFlag := cmd.PersistentFlags().Lookup("image")
		require.NotNil(t, imageFlag, "should have --image flag")
	})
}

func TestWorkflowManager(t *testing.T) {
	wm := &WorkflowManager{
		clusterName: "test-cluster",
		imageName:   "test-image",
		scenario:    constants.ScenarioDevelopment,
		overlay:     "local",
	}
	
	t.Run("workflow manager fields", func(t *testing.T) {
		assert.Equal(t, "test-cluster", wm.clusterName)
		assert.Equal(t, "test-image", wm.imageName)
		assert.Equal(t, constants.ScenarioDevelopment, wm.scenario)
		assert.Equal(t, "local", wm.overlay)
	})
}

func TestWorkflowFullCmd(t *testing.T) {
	wm := &WorkflowManager{
		clusterName: "test-cluster",
		imageName:   "test-image",
		scenario:    constants.ScenarioMinimal,
		overlay:     "local",
	}
	
	cmd := workflowFullCmd(wm)
	
	t.Run("command properties", func(t *testing.T) {
		assert.Equal(t, "full", cmd.Use)
		assert.Contains(t, cmd.Short, "complete workflow")
		assert.NotNil(t, cmd.RunE)
	})
}

func TestWorkflowSetupCmd(t *testing.T) {
	wm := &WorkflowManager{
		clusterName: "test-cluster",
		scenario:    constants.ScenarioDevelopment,
		overlay:     "local",
	}
	
	cmd := workflowSetupCmd(wm)
	
	t.Run("command properties", func(t *testing.T) {
		assert.Equal(t, "setup", cmd.Use)
		assert.Contains(t, cmd.Short, "cluster and deploy infrastructure")
		assert.NotNil(t, cmd.RunE)
	})
}

// Test that scenarios are properly integrated into workflow
func TestWorkflowScenarioIntegration(t *testing.T) {
	scenarios := []string{
		constants.ScenarioMinimal,
		constants.ScenarioDevelopment,
		constants.ScenarioServiceMesh,
		constants.ScenarioMonitoring,
		constants.ScenarioFullStack,
	}
	
	for _, scenario := range scenarios {
		t.Run("scenario_"+scenario, func(t *testing.T) {
			wm := &WorkflowManager{
				clusterName: "test-cluster",
				imageName:   "test-image",
				scenario:    scenario,
				overlay:     "local",
			}
			
			// Verify workflow manager accepts the scenario
			assert.Equal(t, scenario, wm.scenario)
			
			// Verify commands are created without panic
			assert.NotPanics(t, func() {
				_ = workflowFullCmd(wm)
				_ = workflowSetupCmd(wm)
			})
		})
	}
}