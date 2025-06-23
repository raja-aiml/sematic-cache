// Package tools provides unit tests for tool registry
package tools

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
)

func TestRegistry_Register(t *testing.T) {
	tests := []struct {
		name      string
		tool      interfaces.ToolInstaller
		setupMock func(*mockToolInstaller)
		expectErr bool
		errMsg    string
	}{
		{
			name: "register valid tool",
			setupMock: func(m *mockToolInstaller) {
				m.On("Name").Return("test-tool")
			},
			expectErr: false,
		},
		{
			name:      "register nil tool",
			tool:      nil,
			expectErr: true,
			errMsg:    "tool cannot be nil",
		},
		{
			name: "register tool with empty name",
			setupMock: func(m *mockToolInstaller) {
				m.On("Name").Return("")
			},
			expectErr: true,
			errMsg:    "tool name cannot be empty",
		},
		{
			name: "register duplicate tool",
			setupMock: func(m *mockToolInstaller) {
				m.On("Name").Return("test-tool")
			},
			expectErr: true,
			errMsg:    "tool test-tool already registered",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := &mockLogger{}
			registry := NewRegistry(logger)

			// For duplicate test, register first tool
			if tt.name == "register duplicate tool" {
				firstTool := &mockToolInstaller{}
				firstTool.On("Name").Return("test-tool")
				logger.On("Debug", "Registered tool: %s", []interface{}{"test-tool"})
				err := registry.Register(firstTool)
				require.NoError(t, err)
			}

			var tool interfaces.ToolInstaller
			if tt.tool != nil {
				tool = tt.tool
			} else if tt.setupMock != nil {
				mockTool := &mockToolInstaller{}
				tt.setupMock(mockTool)
				tool = mockTool
			}

			if tool != nil && tool.Name() != "" && tt.name != "register duplicate tool" {
				logger.On("Debug", "Registered tool: %s", []interface{}{tool.Name()})
			}

			err := registry.Register(tool)

			if tt.expectErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errMsg)
			} else {
				assert.NoError(t, err)
			}

			logger.AssertExpectations(t)
		})
	}
}

func TestRegistry_Get(t *testing.T) {
	logger := &mockLogger{}
	registry := NewRegistry(logger)

	// Register a tool
	tool := &mockToolInstaller{}
	tool.On("Name").Return("test-tool")
	logger.On("Debug", "Registered tool: %s", []interface{}{"test-tool"})
	err := registry.Register(tool)
	require.NoError(t, err)

	tests := []struct {
		name      string
		toolName  string
		expectErr bool
	}{
		{
			name:      "get existing tool",
			toolName:  "test-tool",
			expectErr: false,
		},
		{
			name:      "get non-existing tool",
			toolName:  "unknown-tool",
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := registry.Get(tt.toolName)

			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				assert.Equal(t, tool, result)
			}
		})
	}
}

func TestRegistry_List(t *testing.T) {
	logger := &mockLogger{}
	registry := NewRegistry(logger)

	// Register multiple tools
	tools := make([]*mockToolInstaller, 3)
	for i := 0; i < 3; i++ {
		tools[i] = &mockToolInstaller{}
		tools[i].On("Name").Return(fmt.Sprintf("tool-%d", i))
		logger.On("Debug", "Registered tool: %s", []interface{}{fmt.Sprintf("tool-%d", i)})
		err := registry.Register(tools[i])
		require.NoError(t, err)
	}

	// List tools
	result := registry.List()
	assert.Len(t, result, 3)

	// Verify all tools are in the list
	toolMap := make(map[string]bool)
	for _, tool := range result {
		toolMap[tool.Name()] = true
	}

	for i := 0; i < 3; i++ {
		assert.True(t, toolMap[fmt.Sprintf("tool-%d", i)])
	}
}

func TestRegistry_InstallAll(t *testing.T) {
	tests := []struct {
		name  string
		tools []struct {
			name       string
			installed  bool
			version    string
			installErr error
		}
		expectErr bool
		errMsg    string
	}{
		{
			name: "all tools already installed",
			tools: []struct {
				name       string
				installed  bool
				version    string
				installErr error
			}{
				{name: "tool-1", installed: true, version: "v1.0.0"},
				{name: "tool-2", installed: true, version: "v2.0.0"},
			},
			expectErr: false,
		},
		{
			name: "install missing tools",
			tools: []struct {
				name       string
				installed  bool
				version    string
				installErr error
			}{
				{name: "tool-1", installed: true, version: "v1.0.0"},
				{name: "tool-2", installed: false},
			},
			expectErr: false,
		},
		{
			name: "installation fails",
			tools: []struct {
				name       string
				installed  bool
				version    string
				installErr error
			}{
				{name: "tool-1", installed: false, installErr: fmt.Errorf("install failed")},
			},
			expectErr: true,
			errMsg:    "failed to install tool-1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := &mockLogger{}
			registry := NewRegistry(logger)
			ctx := context.Background()

			// Setup tools
			var toInstallCount int
			for _, toolConfig := range tt.tools {
				tool := &mockToolInstaller{}
				tool.On("Name").Return(toolConfig.name)
				tool.On("IsInstalled").Return(toolConfig.installed)
				tool.On("Description").Return("Test tool").Maybe()

				if toolConfig.installed {
					tool.On("GetInstalledVersion").Return(toolConfig.version, nil)
					logger.On("Success", "%s is already installed: %s",
						[]interface{}{toolConfig.name, toolConfig.version})
				} else {
					toInstallCount++
					if toolConfig.installErr != nil {
						tool.On("Install", ctx).Return(toolConfig.installErr)
					} else {
						tool.On("Install", ctx).Return(nil)
					}
					logger.On("Info", "Installing %s (%s)...",
						[]interface{}{toolConfig.name, "Test tool"})
				}

				logger.On("Debug", "Registered tool: %s", []interface{}{toolConfig.name})
				err := registry.Register(tool)
				require.NoError(t, err)
			}

			// Setup logger expectations
			logger.On("Info", "Installing %d tools...", []interface{}{len(tt.tools)})

			if toInstallCount == 0 {
				logger.On("Success", "All tools are already installed!", []interface{}(nil))
			} else {
				logger.On("Info", "Installing %d missing tools...", []interface{}{toInstallCount})
				if !tt.expectErr {
					logger.On("Success", "All tools installed successfully!", []interface{}(nil))
				}
			}

			// Execute
			err := registry.InstallAll(ctx)

			// Verify
			if tt.expectErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errMsg)
			} else {
				assert.NoError(t, err)
			}

			logger.AssertExpectations(t)
		})
	}
}

func TestRegistry_ValidateAll(t *testing.T) {
	tests := []struct {
		name  string
		tools []struct {
			name        string
			validateErr error
		}
		expectErr bool
	}{
		{
			name: "all tools valid",
			tools: []struct {
				name        string
				validateErr error
			}{
				{name: "tool-1", validateErr: nil},
				{name: "tool-2", validateErr: nil},
			},
			expectErr: false,
		},
		{
			name: "some tools invalid",
			tools: []struct {
				name        string
				validateErr error
			}{
				{name: "tool-1", validateErr: nil},
				{name: "tool-2", validateErr: fmt.Errorf("validation failed")},
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := &mockLogger{}
			registry := NewRegistry(logger)

			// Setup tools
			for _, toolConfig := range tt.tools {
				tool := &mockToolInstaller{}
				tool.On("Name").Return(toolConfig.name)
				tool.On("Validate").Return(toolConfig.validateErr)

				logger.On("Debug", "Registered tool: %s", []interface{}{toolConfig.name})
				err := registry.Register(tool)
				require.NoError(t, err)
			}

			// Setup logger expectations
			logger.On("Info", "Validating %d tools...", []interface{}{len(tt.tools)})
			if !tt.expectErr {
				logger.On("Success", "All tools validated successfully!", []interface{}(nil))
			}

			// Execute
			err := registry.ValidateAll()

			// Verify
			if tt.expectErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "validation failed")
			} else {
				assert.NoError(t, err)
			}

			logger.AssertExpectations(t)
		})
	}
}

func TestRegistry_InstallAllWithOptions_Sequential(t *testing.T) {
	logger := &mockLogger{}
	registry := NewRegistry(logger)
	ctx := context.Background()

	// Setup tools
	for i := 0; i < 2; i++ {
		tool := &mockToolInstaller{}
		toolName := fmt.Sprintf("tool-%d", i)
		tool.On("Name").Return(toolName)
		tool.On("IsInstalled").Return(false)
		tool.On("Description").Return("Test tool")
		tool.On("Install", ctx).Return(nil)
		tool.On("Validate").Return(nil)

		logger.On("Debug", "Registered tool: %s", []interface{}{toolName})
		logger.On("Info", "Installing %s (%s)...", []interface{}{toolName, "Test tool"})

		err := registry.Register(tool)
		require.NoError(t, err)
	}

	logger.On("Info", "Installing %d tools...", []interface{}{2})

	// Execute with sequential options (no parallel)
	opts := interfaces.InstallOptions{
		Parallel:       false,
		MaxConcurrency: 1,
		Force:          false,
		SkipValidation: false,
	}

	err := registry.InstallAllWithOptions(ctx, opts)
	assert.NoError(t, err)

	logger.AssertExpectations(t)
}

func TestRegistry_InstallAllWithOptions_Parallel(t *testing.T) {
	logger := &mockLogger{}
	registry := NewRegistry(logger)
	ctx := context.Background()

	// Setup tools
	for i := 0; i < 3; i++ {
		tool := &mockToolInstaller{}
		toolName := fmt.Sprintf("tool-%d", i)
		tool.On("Name").Return(toolName)
		tool.On("IsInstalled").Return(false)
		tool.On("Description").Return("Test tool")
		tool.On("Install", ctx).Return(nil).After(10 * time.Millisecond) // Simulate work
		tool.On("Validate").Return(nil)

		logger.On("Debug", "Registered tool: %s", []interface{}{toolName})
		logger.On("Info", "Installing %s (%s)...", []interface{}{toolName, "Test tool"})

		err := registry.Register(tool)
		require.NoError(t, err)
	}

	logger.On("Info", "Installing %d tools...", []interface{}{3})
	logger.On("Success", "All tools installed successfully!", []interface{}(nil))

	// Execute with parallel options
	opts := interfaces.InstallOptions{
		Parallel:       true,
		MaxConcurrency: 2,
		Force:          false,
		SkipValidation: false,
	}

	start := time.Now()
	err := registry.InstallAllWithOptions(ctx, opts)
	duration := time.Since(start)

	// Verify
	assert.NoError(t, err)
	// With concurrency of 2 and 3 tools, it should take at least 20ms
	// (2 tools in parallel, then 1 more)
	assert.GreaterOrEqual(t, duration, 20*time.Millisecond)

	logger.AssertExpectations(t)
}
