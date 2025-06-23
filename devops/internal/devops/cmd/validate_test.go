package cmd

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateCommand(t *testing.T) {
	// Create temp directory for test files
	tempDir, err := os.MkdirTemp("", "validate-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	tests := []struct {
		name       string
		files      map[string]string
		args       []string
		wantErr    bool
		wantOutput []string
	}{
		{
			name: "valid taskfile",
			files: map[string]string{
				"Taskfile.yaml": `version: '3'
tasks:
  test:
    desc: Test task
    cmds:
      - echo "test"
`,
			},
			args:       []string{}, // Empty args will search current directory
			wantErr:    false,
			wantOutput: []string{"✅ All 1 Taskfiles are valid"},
		},
		{
			name: "invalid yaml",
			files: map[string]string{
				"Taskfile.yaml": `version: '3'
tasks:
  test:
    desc: Test task
    cmds:
      - echo: "invalid syntax"
        this is not valid yaml
`,
			},
			args:       []string{},
			wantErr:    true,
			wantOutput: []string{"❌ Error in", "Taskfile.yaml"},
		},
		{
			name: "missing version",
			files: map[string]string{
				"Taskfile.yaml": `tasks:
  test:
    desc: Test task
    cmds:
      - echo "test"
`,
			},
			args:       []string{},
			wantErr:    true,
			wantOutput: []string{"Missing required field: version"},
		},
		{
			name: "multiple taskfiles",
			files: map[string]string{
				"Taskfile.yaml": `version: '3'
tasks:
  test:
    desc: Test task
    cmds:
      - echo "test"
`,
				"sub/Taskfile.yaml": `version: '3'
tasks:
  build:
    desc: Build task
    cmds:
      - go build
`,
			},
			args:       []string{},
			wantErr:    false,
			wantOutput: []string{"✅ All 2 Taskfiles are valid"},
		},
		{
			name: "verbose output",
			files: map[string]string{
				"Taskfile.yaml": `version: '3'
tasks:
  test:
    desc: Test task
    cmds:
      - echo "test"
`,
			},
			args:       []string{"--verbose"},
			wantErr:    false,
			wantOutput: []string{"✅", "Taskfile.yaml"},
		},
		{
			name:       "no taskfiles",
			files:      map[string]string{},
			args:       []string{},
			wantErr:    false,
			wantOutput: []string{"✅ All 0 Taskfiles are valid"},
		},
		{
			name: "taskfile without tasks",
			files: map[string]string{
				"Taskfile.yaml": `version: '3'
`,
			},
			args:       []string{},
			wantErr:    false,
			wantOutput: []string{"✅ All 1 Taskfiles are valid"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create new temp dir for each test
			testDir, err := os.MkdirTemp("", "validate-test-*")
			require.NoError(t, err)
			defer os.RemoveAll(testDir)

			// Create test files
			var createdFiles []string
			for path, content := range tt.files {
				fullPath := filepath.Join(testDir, path)
				dir := filepath.Dir(fullPath)
				err := os.MkdirAll(dir, 0755)
				require.NoError(t, err)
				err = os.WriteFile(fullPath, []byte(content), 0644)
				require.NoError(t, err)
				createdFiles = append(createdFiles, fullPath)
			}

			// Set command args - if no args provided, use the created files
			args := tt.args
			if len(args) == 0 && len(createdFiles) > 0 {
				args = createdFiles
			}
			cmdArgs := append([]string{"validate"}, args...)
			rootCmd.SetArgs(cmdArgs)

			// Capture output
			var buf bytes.Buffer
			rootCmd.SetOut(&buf)
			rootCmd.SetErr(&buf)

			// Execute command
			err = rootCmd.Execute()

			// Check error
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}

			// Check output
			output := buf.String()
			for _, expected := range tt.wantOutput {
				assert.Contains(t, output, expected)
			}
		})
	}
}

func TestValidateCommandHelp(t *testing.T) {
	// Execute through root command for proper initialization
	rootCmd.SetArgs([]string{"validate", "--help"})

	var buf bytes.Buffer
	rootCmd.SetOut(&buf)
	rootCmd.SetErr(&buf)

	err := rootCmd.Execute()
	assert.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "Validate the syntax of Taskfile")
	assert.Contains(t, output, "validate")
	assert.Contains(t, output, "--verbose")
}
