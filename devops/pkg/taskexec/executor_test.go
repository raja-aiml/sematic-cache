package taskexec

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
	"github.com/raja-aiml/sematic-cache/devops/pkg/logger"
)

func TestExecutor_ValidateDirectory(t *testing.T) {
	tests := []struct {
		name      string
		dir       string
		setupFunc func(string) error
		wantErr   bool
		errMsg    string
	}{
		{
			name:    "nonexistent directory",
			dir:     "/nonexistent/path",
			wantErr: true,
			errMsg:  "directory does not exist",
		},
		{
			name: "directory without devops/tasks",
			dir:  "",
			setupFunc: func(dir string) error {
				return os.MkdirAll(dir, 0755)
			},
			wantErr: true,
			errMsg:  "tasks directory not found",
		},
		{
			name: "valid directory structure",
			dir:  "",
			setupFunc: func(dir string) error {
				tasksDir := filepath.Join(dir, "devops", "tasks", "build")
				return os.MkdirAll(tasksDir, 0755)
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var testDir string
			if tt.setupFunc != nil {
				var err error
				testDir, err = os.MkdirTemp("", "taskexec-test-*")
				require.NoError(t, err)
				defer os.RemoveAll(testDir)

				err = tt.setupFunc(testDir)
				require.NoError(t, err)
			} else {
				testDir = tt.dir
			}

			executor := NewExecutor(logger.NewWithOptions(logger.DebugLevel, false))
			err := executor.ValidateDirectory(testDir)

			if tt.wantErr {
				assert.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestExecutor_LoadTaskfile(t *testing.T) {
	// Create a temporary taskfile
	tmpDir, err := os.MkdirTemp("", "taskexec-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	taskfileContent := `version: '3'

tasks:
  test:
    desc: Test task
    cmds:
      - echo "Hello World"
`

	taskfilePath := filepath.Join(tmpDir, "Taskfile.yaml")
	err = os.WriteFile(taskfilePath, []byte(taskfileContent), 0644)
	require.NoError(t, err)

	executor := NewExecutor(logger.NewWithOptions(logger.DebugLevel, false))

	tests := []struct {
		name    string
		path    string
		wantErr bool
		errMsg  string
	}{
		{
			name:    "valid taskfile",
			path:    taskfilePath,
			wantErr: false,
		},
		{
			name:    "nonexistent taskfile",
			path:    "/nonexistent/Taskfile.yaml",
			wantErr: true,
			errMsg:  "taskfile not found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := executor.LoadTaskfile(tt.path)

			if tt.wantErr {
				assert.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, executor.taskfile)
			}
		})
	}
}

func TestExecutor_ListTasks(t *testing.T) {
	// Create a temporary taskfile
	tmpDir, err := os.MkdirTemp("", "taskexec-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	taskfileContent := `version: '3'

tasks:
  build:
    desc: Build the application
    summary: Builds the Go application
    vars:
      BINARY_NAME: myapp
    cmds:
      - go build -o {{.BINARY_NAME}}

  test:
    desc: Run tests
    deps: [build]
    cmds:
      - go test ./...
`

	taskfilePath := filepath.Join(tmpDir, "Taskfile.yaml")
	err = os.WriteFile(taskfilePath, []byte(taskfileContent), 0644)
	require.NoError(t, err)

	executor := NewExecutor(logger.NewWithOptions(logger.DebugLevel, false))
	err = executor.LoadTaskfile(taskfilePath)
	require.NoError(t, err)

	tasks, err := executor.ListTasks()
	require.NoError(t, err)
	assert.Len(t, tasks, 2)

	// Find build task
	var buildTask *interfaces.TaskInfo
	for i, task := range tasks {
		if task.Name == "build" {
			buildTask = &tasks[i]
			break
		}
	}
	require.NotNil(t, buildTask)

	// Verify build task properties
	assert.Equal(t, "build", buildTask.Name)
	assert.Equal(t, "Build the application", buildTask.Description)
	assert.Equal(t, "Builds the Go application", buildTask.Summary)
}

func TestExecutor_ExecuteTask_NoTaskfileLoaded(t *testing.T) {
	executor := NewExecutor(logger.NewWithOptions(logger.DebugLevel, false))

	err := executor.ExecuteTask(context.Background(), "test", nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no taskfile loaded")
}
