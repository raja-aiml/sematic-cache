package utils

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestFindProjectRoot(t *testing.T) {
	// Test finding project root
	root, err := FindProjectRoot()
	if err != nil {
		t.Errorf("FindProjectRoot() error = %v", err)
		return
	}
	
	// Verify go.mod exists in the root
	goModPath := filepath.Join(root, "go.mod")
	if _, err := os.Stat(goModPath); os.IsNotExist(err) {
		t.Errorf("go.mod not found at %s", goModPath)
	}
}

func TestRunCommand(t *testing.T) {
	tests := []struct {
		name    string
		cmd     string
		args    []string
		opts    *ExecOptions
		wantErr bool
	}{
		{
			name:    "echo command",
			cmd:     "echo",
			args:    []string{"hello"},
			opts:    nil,
			wantErr: false,
		},
		{
			name:    "invalid command",
			cmd:     "invalidcommandthatdoesnotexist",
			args:    []string{},
			opts:    nil,
			wantErr: true,
		},
		{
			name:    "with timeout",
			cmd:     "sleep",
			args:    []string{"5"},
			opts:    &ExecOptions{Timeout: 100 * time.Millisecond},
			wantErr: true,
		},
	}

	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := RunCommand(ctx, tt.cmd, tt.args, tt.opts)
			if (err != nil) != tt.wantErr {
				t.Errorf("RunCommand() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && output == "" {
				t.Errorf("RunCommand() output is empty")
			}
		})
	}
}

func TestExecOptions(t *testing.T) {
	tests := []struct {
		name string
		opts *ExecOptions
	}{
		{
			name: "nil options",
			opts: nil,
		},
		{
			name: "with directory",
			opts: &ExecOptions{
				Dir: "/tmp",
			},
		},
		{
			name: "with environment",
			opts: &ExecOptions{
				Env: []string{"TEST=1"},
			},
		},
		{
			name: "silent mode",
			opts: &ExecOptions{
				Silent: true,
			},
		},
	}

	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test with echo command which should work on all platforms
			_, err := RunCommand(ctx, "echo", []string{"test"}, tt.opts)
			if err != nil {
				t.Errorf("RunCommand() with options %+v failed: %v", tt.opts, err)
			}
		})
	}
}

func TestRunCommandStreaming(t *testing.T) {
	// Test the RunCommandStreaming function if it exists
	// This is a placeholder for when streaming is implemented
	t.Skip("RunCommandStreaming not yet implemented")
}

func TestLogger(t *testing.T) {
	// Test logger creation
	logger := NewLogger("test")
	
	// Test logging methods (just ensure they don't panic)
	logger.Info("test info message")
	logger.Debug("test debug message")
	logger.Warn("test warning message")
	logger.Error("test error message")
	
	// If we get here without panics, the test passes
}