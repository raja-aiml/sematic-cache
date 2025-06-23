package cmd

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRootCommand(t *testing.T) {
	tests := []struct {
		name     string
		args     []string
		wantErr  bool
		contains []string
	}{
		{
			name:     "help command",
			args:     []string{"--help"},
			wantErr:  false,
			contains: []string{"DevOps provides various development", "Available Commands", "taskdoc", "validate", "version"},
		},
		{
			name:     "no args shows help",
			args:     []string{},
			wantErr:  false,
			contains: []string{"DevOps provides various development"},
		},
		{
			name:     "invalid command",
			args:     []string{"invalid"},
			wantErr:  true,
			contains: []string{"unknown command"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Reset command for testing
			rootCmd.SetArgs(tt.args)

			// Capture output
			var buf bytes.Buffer
			rootCmd.SetOut(&buf)
			rootCmd.SetErr(&buf)

			// Execute command
			err := rootCmd.Execute()

			// Check error
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}

			// Check output contains expected strings
			output := buf.String()
			for _, expected := range tt.contains {
				assert.Contains(t, output, expected)
			}
		})
	}
}

func TestExecute(t *testing.T) {
	// Test that Execute function works
	// This is mainly for coverage since it's just a wrapper
	originalArgs := rootCmd.Args
	rootCmd.SetArgs([]string{"--help"})

	// Should not panic
	assert.NotPanics(t, func() {
		// We can't test Execute directly as it calls os.Exit
		// But we can ensure the root command is properly set up
		assert.NotNil(t, rootCmd)
		assert.Equal(t, "devops", rootCmd.Use)
	})

	rootCmd.Args = originalArgs
}
