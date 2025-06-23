package cmd

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTaskdocCommand(t *testing.T) {
	// Create test Taskfile content
	taskfileContent := `version: '3'

vars:
  APP_NAME: test-app
  VERSION: 1.0.0

tasks:
  build:
    desc: Build the application
    deps: [clean]
    cmds:
      - go build -o {{.APP_NAME}}

  test:
    desc: Run tests
    cmds:
      - go test ./...

  clean:
    desc: Clean build artifacts
    cmds:
      - rm -f {{.APP_NAME}}
`

	tests := []struct {
		name       string
		args       []string
		wantErr    bool
		wantOutput []string
	}{
		{
			name:    "generate markdown",
			args:    []string{"--root", "tempDir", "--format", "markdown", "--output", "-"},
			wantErr: false,
			wantOutput: []string{
				"# Taskfile Structure and Dependencies",
				"build",
				"test",
				"clean",
			},
		},
		{
			name:    "generate json",
			args:    []string{"--root", "tempDir", "--format", "json", "--output", "-"},
			wantErr: false,
			wantOutput: []string{
				`"build"`,
				`"Desc": "Build the application"`,
				`"test"`,
				`"clean"`,
			},
		},
		{
			name:    "generate flow",
			args:    []string{"--root", "tempDir", "--flow", "--output", "-"},
			wantErr: false,
			wantOutput: []string{
				"graph TD",
				"build --> clean",
			},
		},
		{
			name:    "verbose output",
			args:    []string{"--root", "tempDir", "--verbose", "--output", "-"},
			wantErr: false,
			wantOutput: []string{
				"# Taskfile Structure",
			},
		},
		{
			name:       "output to file",
			args:       []string{"--root", "tempDir", "--output", filepath.Join("tempDir", "docs.md")},
			wantErr:    false,
			wantOutput: []string{"📄 Documentation generated:"},
		},
		{
			name:       "invalid format",
			args:       []string{"--root", "tempDir", "--format", "invalid", "--output", "-"},
			wantErr:    true,
			wantOutput: []string{"unsupported format"},
		},
		{
			name:       "nonexistent directory",
			args:       []string{"/nonexistent/path"},
			wantErr:    true,
			wantOutput: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create temp directory for this test
			tempDir, err := os.MkdirTemp("", "taskdoc-test-*")
			require.NoError(t, err)
			defer os.RemoveAll(tempDir)

			// Create test Taskfile
			taskfilePath := filepath.Join(tempDir, "Taskfile.yaml")
			err = os.WriteFile(taskfilePath, []byte(taskfileContent), 0644)
			require.NoError(t, err)

			// Replace tempDir placeholder in args
			args := make([]string, len(tt.args))
			copy(args, tt.args)
			for i := 0; i < len(args); i++ {
				if args[i] == "--root" && i+1 < len(args) && args[i+1] == "tempDir" {
					args[i+1] = tempDir
				} else if args[i] == "--output" && i+1 < len(args) && strings.Contains(args[i+1], "tempDir") {
					args[i+1] = strings.Replace(args[i+1], "tempDir", tempDir, 1)
				}
			}

			// Set command args
			cmdArgs := append([]string{"taskdoc"}, args...)
			rootCmd.SetArgs(cmdArgs)

			// Capture output
			var buf bytes.Buffer
			rootCmd.SetOut(&buf)
			rootCmd.SetErr(&buf)

			// Execute command
			err = rootCmd.Execute()

			// Check error
			if tt.wantErr {
				assert.Error(t, err, "Expected error but got none")
			} else {
				assert.NoError(t, err, "Unexpected error: %v", err)
			}

			// Check output
			output := buf.String()
			for _, expected := range tt.wantOutput {
				assert.Contains(t, output, expected)
			}

			// Check if output file was created
			for i, arg := range args {
				if arg == "--output" && i+1 < len(args) {
					outputFile := args[i+1]
					if outputFile != "-" && !tt.wantErr && !strings.Contains(outputFile, "tempDir") {
						// Check if file was created
						if _, err := os.Stat(outputFile); err == nil {
							os.Remove(outputFile)
						}
					}
				}
			}
		})
	}
}

func TestTaskdocCommandHelp(t *testing.T) {
	// Execute through root command for proper initialization
	rootCmd.SetArgs([]string{"taskdoc", "--help"})

	var buf bytes.Buffer
	rootCmd.SetOut(&buf)
	rootCmd.SetErr(&buf)

	err := rootCmd.Execute()
	assert.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "Generate comprehensive documentation")
	assert.Contains(t, output, "taskdoc")
	assert.Contains(t, output, "--format")
	assert.Contains(t, output, "--output")
	assert.Contains(t, output, "--verbose")
}

func TestTaskdocEmptyDirectory(t *testing.T) {
	// Create empty temp directory
	tempDir, err := os.MkdirTemp("", "taskdoc-empty-*")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	rootCmd.SetArgs([]string{"taskdoc", "--root", tempDir, "--output", "-"})

	var buf bytes.Buffer
	rootCmd.SetOut(&buf)
	rootCmd.SetErr(&buf)

	err = rootCmd.Execute()
	assert.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "No Taskfiles found")
}

func TestTaskdocMultipleTaskfiles(t *testing.T) {
	// Create temp directory with multiple Taskfiles
	tempDir, err := os.MkdirTemp("", "taskdoc-multi-*")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create root Taskfile
	rootTaskfile := `version: '3'
tasks:
  root-task:
    desc: Root task
    cmds:
      - echo "root"
`
	err = os.WriteFile(filepath.Join(tempDir, "Taskfile.yaml"), []byte(rootTaskfile), 0644)
	require.NoError(t, err)

	// Create subdirectory with Taskfile
	subDir := filepath.Join(tempDir, "sub")
	err = os.MkdirAll(subDir, 0755)
	require.NoError(t, err)

	subTaskfile := `version: '3'
tasks:
  sub-task:
    desc: Sub task
    cmds:
      - echo "sub"
`
	err = os.WriteFile(filepath.Join(subDir, "Taskfile.yaml"), []byte(subTaskfile), 0644)
	require.NoError(t, err)

	rootCmd.SetArgs([]string{"taskdoc", "--root", tempDir, "--output", "-"})

	var buf bytes.Buffer
	rootCmd.SetOut(&buf)
	rootCmd.SetErr(&buf)

	err = rootCmd.Execute()
	assert.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "root-task")
	assert.Contains(t, output, "sub-task")
	assert.Contains(t, output, "sub/Taskfile.yaml")
}

func indexOf(slice []string, item string) int {
	for i, v := range slice {
		if v == item {
			return i
		}
	}
	return -1
}
