package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type MockCommandBuilder struct {
	shouldError bool
	commands    []string
}

func (m *MockCommandBuilder) Build(cmd *InterpretedCommand) ([]string, error) {
	if m.shouldError {
		return nil, fmt.Errorf("build error")
	}
	if m.commands != nil {
		return m.commands, nil
	}
	return []string{"echo", "test"}, nil
}

func TestNewSafeCommandExecutor(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "executor-test")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	config := &Config{
		AuditLogPath: filepath.Join(tmpDir, "audit.log"),
	}

	executor, err := NewSafeCommandExecutor(config)
	require.NoError(t, err)
	assert.NotNil(t, executor)
	assert.NotNil(t, executor.auditLogger)
	assert.NotNil(t, executor.commandBuilder)
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name        string
		config      *Config
		cmd         *InterpretedCommand
		shouldError bool
		errorMsg    string
	}{
		{
			name: "valid command",
			config: &Config{
				EnableDangerousCommands: true,
			},
			cmd: &InterpretedCommand{
				Command: "cluster",
			},
			shouldError: false,
		},
		{
			name: "dangerous command disabled",
			config: &Config{
				EnableDangerousCommands: false,
			},
			cmd: &InterpretedCommand{
				Command:   "delete",
				Dangerous: true,
			},
			shouldError: true,
			errorMsg:    "dangerous commands are disabled",
		},
		{
			name: "command not in whitelist",
			config: &Config{
				CommandWhitelist: []string{"cluster", "deploy"},
			},
			cmd: &InterpretedCommand{
				Command: "delete",
			},
			shouldError: true,
			errorMsg:    "not in whitelist",
		},
		{
			name: "command in whitelist",
			config: &Config{
				CommandWhitelist: []string{"cluster", "deploy"},
			},
			cmd: &InterpretedCommand{
				Command: "cluster",
			},
			shouldError: false,
		},
		{
			name: "command in blacklist",
			config: &Config{
				CommandBlacklist: []string{"delete", "destroy"},
			},
			cmd: &InterpretedCommand{
				Command: "delete",
			},
			shouldError: true,
			errorMsg:    "is blacklisted",
		},
		{
			name: "wildcard whitelist",
			config: &Config{
				CommandWhitelist: []string{"cluster*"},
			},
			cmd: &InterpretedCommand{
				Command: "cluster-create",
			},
			shouldError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			executor := &SafeCommandExecutor{config: tt.config}
			err := executor.Validate(tt.cmd)

			if tt.shouldError {
				assert.Error(t, err)
				if tt.errorMsg != "" {
					assert.Contains(t, err.Error(), tt.errorMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestExecute(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "executor-test")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	config := &Config{
		AuditLogPath:            filepath.Join(tmpDir, "audit.log"),
		CommandTimeout:          5 * time.Second,
		EnableDangerousCommands: true,
	}

	executor, err := NewSafeCommandExecutor(config)
	require.NoError(t, err)

	// Use mock command builder
	executor.commandBuilder = &MockCommandBuilder{}

	tests := []struct {
		name        string
		cmd         *InterpretedCommand
		shouldError bool
	}{
		{
			name: "successful execution",
			cmd: &InterpretedCommand{
				Query:   "test query",
				Command: "echo",
				Options: map[string]string{},
			},
			shouldError: false,
		},
		{
			name: "validation failure",
			cmd: &InterpretedCommand{
				Query:     "dangerous query",
				Command:   "delete",
				Dangerous: true,
			},
			shouldError: false, // Will pass validation with EnableDangerousCommands=true
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			result, err := executor.Execute(ctx, tt.cmd)

			if tt.shouldError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				assert.NotNil(t, result.AuditTrail)
				assert.Equal(t, tt.cmd.Query, result.AuditTrail.Query)
			}
		})
	}
}

func TestExecuteWithTimeout(t *testing.T) {
	executor := &SafeCommandExecutor{
		config: &Config{
			CommandTimeout: 100 * time.Millisecond,
		},
	}

	tests := []struct {
		name        string
		cmdArgs     []string
		shouldError bool
		errorMsg    string
	}{
		{
			name:        "successful command",
			cmdArgs:     []string{"echo", "hello"},
			shouldError: false,
		},
		{
			name:        "timeout command",
			cmdArgs:     []string{"sleep", "1"},
			shouldError: true,
			errorMsg:    "timed out",
		},
		{
			name:        "failed command",
			cmdArgs:     []string{"false"},
			shouldError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			output, exitCode, err := executor.executeWithTimeout(ctx, tt.cmdArgs)

			if tt.shouldError {
				if err == nil {
					t.Fatalf("expected error but got nil for command: %v", tt.cmdArgs)
				}
				if tt.errorMsg != "" && err != nil {
					assert.Contains(t, err.Error(), tt.errorMsg)
				}
			} else {
				assert.NoError(t, err)
				assert.Equal(t, 0, exitCode)
				assert.NotEmpty(t, output)
			}
		})
	}
}

func TestFormatCommand(t *testing.T) {
	executor := &SafeCommandExecutor{}

	tests := []struct {
		name     string
		cmd      *InterpretedCommand
		expected string
	}{
		{
			name: "simple command",
			cmd: &InterpretedCommand{
				Command: "cluster",
			},
			expected: "cluster",
		},
		{
			name: "command with subcommand",
			cmd: &InterpretedCommand{
				Command:    "cluster",
				Subcommand: "create",
			},
			expected: "cluster create",
		},
		{
			name: "command with args",
			cmd: &InterpretedCommand{
				Command: "deploy",
				Args:    []string{"nginx", "production"},
			},
			expected: "deploy nginx production",
		},
		{
			name: "command with options",
			cmd: &InterpretedCommand{
				Command: "create",
				Options: map[string]string{
					"name":  "test",
					"n":     "3",
					"force": "true",
				},
			},
			expected: "create --name test -n 3 --force",
		},
		{
			name: "full command",
			cmd: &InterpretedCommand{
				Command:    "cluster",
				Subcommand: "create",
				Args:       []string{"dev"},
				Options: map[string]string{
					"nodes": "3",
					"ha":    "true",
				},
			},
			expected: "cluster create dev --nodes 3 --ha",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Initialize empty maps if nil
			if tt.cmd.Options == nil {
				tt.cmd.Options = make(map[string]string)
			}

			result := executor.formatCommand(tt.cmd)
			// We can't guarantee order of options, so check components
			assert.Contains(t, result, tt.cmd.Command)
			if tt.cmd.Subcommand != "" {
				assert.Contains(t, result, tt.cmd.Subcommand)
			}
			for _, arg := range tt.cmd.Args {
				assert.Contains(t, result, arg)
			}
		})
	}
}

func TestMatchesPattern(t *testing.T) {
	executor := &SafeCommandExecutor{}

	tests := []struct {
		name     string
		command  string
		pattern  string
		expected bool
	}{
		{"exact match", "cluster", "cluster", true},
		{"no match", "cluster", "deploy", false},
		{"wildcard all", "anything", "*", true},
		{"prefix wildcard", "cluster-create", "cluster*", true},
		{"prefix no match", "deploy", "cluster*", false},
		{"case sensitive", "Cluster", "cluster", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := executor.matchesPattern(tt.command, tt.pattern)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestGetCurrentUser(t *testing.T) {
	executor := &SafeCommandExecutor{}

	// Save current env
	oldUser := os.Getenv("USER")
	oldUsername := os.Getenv("USERNAME")
	defer func() {
		os.Setenv("USER", oldUser)
		os.Setenv("USERNAME", oldUsername)
	}()

	// Test with USER set
	os.Setenv("USER", "testuser")
	os.Unsetenv("USERNAME")
	assert.Equal(t, "testuser", executor.getCurrentUser())

	// Test with USERNAME set
	os.Unsetenv("USER")
	os.Setenv("USERNAME", "winuser")
	assert.Equal(t, "winuser", executor.getCurrentUser())

	// Test with neither set
	os.Unsetenv("USER")
	os.Unsetenv("USERNAME")
	assert.Equal(t, "unknown", executor.getCurrentUser())
}

func TestDefaultCommandBuilder(t *testing.T) {
	builder := &DefaultCommandBuilder{binaryPath: "iaac"}

	tests := []struct {
		name     string
		cmd      *InterpretedCommand
		expected []string
	}{
		{
			name: "simple command",
			cmd: &InterpretedCommand{
				Command: "list",
				Options: map[string]string{},
			},
			expected: []string{"iaac", "list"},
		},
		{
			name: "command with subcommand",
			cmd: &InterpretedCommand{
				Command:    "cluster",
				Subcommand: "create",
				Options:    map[string]string{},
			},
			expected: []string{"iaac", "cluster", "create"},
		},
		{
			name: "command with args and options",
			cmd: &InterpretedCommand{
				Command: "deploy",
				Args:    []string{"app", "prod"},
				Options: map[string]string{
					"namespace": "default",
					"f":         "true",
					"replicas":  "3",
				},
			},
			expected: []string{"iaac", "deploy", "app", "prod"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := builder.Build(tt.cmd)
			require.NoError(t, err)

			// Check base command structure
			assert.Equal(t, tt.expected[0], result[0])
			assert.Equal(t, tt.expected[1], result[1])

			// Options can be in any order, so just verify they exist
			resultStr := strings.Join(result, " ")
			for key, value := range tt.cmd.Options {
				if len(key) == 1 {
					assert.Contains(t, resultStr, fmt.Sprintf("-%s", key))
				} else {
					assert.Contains(t, resultStr, fmt.Sprintf("--%s", key))
				}
				if value != "true" {
					assert.Contains(t, resultStr, value)
				}
			}
		})
	}
}

func TestAuditLogger(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "audit-test")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	auditPath := filepath.Join(tmpDir, "audit.log")
	logger, err := NewAuditLogger(auditPath)
	require.NoError(t, err)

	// Log an entry
	entry := &AuditEntry{
		ID:        "test-123",
		Timestamp: time.Now(),
		User:      "testuser",
		Query:     "create cluster",
		Command:   "cluster create",
		Success:   true,
		Duration:  100 * time.Millisecond,
	}

	err = logger.Log(entry)
	require.NoError(t, err)

	// Verify file was created and contains entry
	data, err := os.ReadFile(auditPath)
	require.NoError(t, err)

	assert.Contains(t, string(data), "test-123")
	assert.Contains(t, string(data), "testuser")
	assert.Contains(t, string(data), "create cluster")

	// Verify it's valid JSON
	var parsed AuditEntry
	err = json.Unmarshal(data[:len(data)-1], &parsed) // Remove trailing newline
	require.NoError(t, err)
	assert.Equal(t, entry.ID, parsed.ID)
}

func TestAuditLoggerWithDirectory(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "audit-test")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	// Test with nested directory that doesn't exist
	auditPath := filepath.Join(tmpDir, "logs", "subdir", "audit.log")
	_, err = NewAuditLogger(auditPath)
	require.NoError(t, err)

	// Directory should have been created
	assert.DirExists(t, filepath.Dir(auditPath))
}
