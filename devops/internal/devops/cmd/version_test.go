package cmd

import (
	"bytes"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVersionCommand(t *testing.T) {
	tests := []struct {
		name       string
		version    string
		gitCommit  string
		buildTime  string
		wantOutput []string
	}{
		{
			name:      "default version info",
			version:   "dev",
			gitCommit: "none",
			buildTime: "unknown",
			wantOutput: []string{
				"DevOps CLI Tool",
				"Version:    dev",
				"Git Commit: none",
				"Build Time: unknown",
				"Go Version: " + runtime.Version(),
				"Platform:   " + runtime.GOOS + "/" + runtime.GOARCH,
			},
		},
		{
			name:      "custom version info",
			version:   "1.0.0",
			gitCommit: "abc123",
			buildTime: "2024-01-01T00:00:00Z",
			wantOutput: []string{
				"DevOps CLI Tool",
				"Version:    1.0.0",
				"Git Commit: abc123",
				"Build Time: 2024-01-01T00:00:00Z",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			origVersion := Version
			origCommit := Commit
			origDate := Date

			// Set test values
			Version = tt.version
			Commit = tt.gitCommit
			Date = tt.buildTime

			// Reset after test
			defer func() {
				Version = origVersion
				Commit = origCommit
				Date = origDate
			}()

			// Execute command through root
			rootCmd.SetArgs([]string{"version"})

			// Capture output
			var buf bytes.Buffer
			rootCmd.SetOut(&buf)
			rootCmd.SetErr(&buf)

			// Execute command
			err := rootCmd.Execute()
			assert.NoError(t, err)

			// Check output
			output := buf.String()
			for _, expected := range tt.wantOutput {
				assert.Contains(t, output, expected)
			}
		})
	}
}

func TestVersionCommandIntegration(t *testing.T) {
	// Test version command through root command
	rootCmd.SetArgs([]string{"version"})

	var buf bytes.Buffer
	rootCmd.SetOut(&buf)
	rootCmd.SetErr(&buf)

	err := rootCmd.Execute()
	assert.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "DevOps CLI Tool")
	assert.Contains(t, output, "Version:")
}
