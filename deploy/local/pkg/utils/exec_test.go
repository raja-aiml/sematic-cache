package utils

import (
	"context"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestRunCommand(t *testing.T) {
	tests := []struct {
		name    string
		cmd     string
		args    []string
		opts    *ExecOptions
		want    string
		wantErr bool
	}{
		{
			name:    "echo_command",
			cmd:     "echo",
			args:    []string{"hello"},
			opts:    nil,
			want:    "hello\n",
			wantErr: false,
		},
		{
			name:    "echo_with_multiple_args",
			cmd:     "echo",
			args:    []string{"hello", "world"},
			opts:    nil,
			want:    "hello world\n",
			wantErr: false,
		},
		{
			name:    "invalid_command",
			cmd:     "invalidcommandthatdoesnotexist",
			args:    []string{},
			opts:    nil,
			want:    "",
			wantErr: true,
		},
		{
			name: "command_with_timeout",
			cmd:  "sleep",
			args: []string{"0.1"},
			opts: &ExecOptions{
				Timeout: 200 * time.Millisecond,
			},
			want:    "",
			wantErr: false,
		},
		{
			name: "command_timeout_exceeded",
			cmd:  "sleep",
			args: []string{"2"},
			opts: &ExecOptions{
				Timeout: 100 * time.Millisecond,
			},
			want:    "",
			wantErr: true,
		},
		{
			name: "command_with_env",
			cmd:  "sh",
			args: []string{"-c", "echo $TEST_ENV_VAR"},
			opts: &ExecOptions{
				Env: []string{"TEST_ENV_VAR=test_value"},
			},
			want:    "test_value\n",
			wantErr: false,
		},
		{
			name: "command_with_dir",
			cmd:  "pwd",
			args: []string{},
			opts: &ExecOptions{
				Dir: "/tmp",
			},
			want:    "/tmp\n",
			wantErr: false,
		},
		{
			name: "silent_mode",
			cmd:  "echo",
			args: []string{"silent"},
			opts: &ExecOptions{
				Silent: true,
			},
			want:    "silent\n",
			wantErr: false,
		},
		{
			name:    "command_with_stderr",
			cmd:     "sh",
			args:    []string{"-c", "echo error >&2; exit 1"},
			opts:    nil,
			want:    "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Skip pwd test on Windows
			if tt.name == "command_with_dir" && runtime.GOOS == "windows" {
				t.Skip("Skipping pwd test on Windows")
			}

			ctx := context.Background()
			got, err := RunCommand(ctx, tt.cmd, tt.args, tt.opts)

			if (err != nil) != tt.wantErr {
				t.Errorf("RunCommand() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			// Special handling for pwd command
			if tt.name == "command_with_dir" && !tt.wantErr {
				// Resolve symlinks for comparison
				gotPath := strings.TrimSpace(got)
				wantPath := strings.TrimSpace(tt.want)
				gotResolved, _ := filepath.EvalSymlinks(gotPath)
				wantResolved, _ := filepath.EvalSymlinks(wantPath)

				if gotResolved != wantResolved {
					t.Errorf("RunCommand() = %v, want %v", got, tt.want)
				}
			} else if got != tt.want {
				t.Errorf("RunCommand() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRunCommand_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	// Start a long-running command
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	_, err := RunCommand(ctx, "sleep", []string{"2"}, nil)
	if err == nil {
		t.Error("RunCommand() expected error for cancelled context")
	}
}

func TestCommandExists(t *testing.T) {
	tests := []struct {
		name     string
		command  string
		expected bool
	}{
		{
			name:     "echo_exists",
			command:  "echo",
			expected: true,
		},
		{
			name:     "sh_exists",
			command:  "sh",
			expected: true,
		},
		{
			name:     "nonexistent_command",
			command:  "thiscommanddoesnotexist",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CommandExists(tt.command)
			if got != tt.expected {
				t.Errorf("CommandExists(%s) = %v, want %v", tt.command, got, tt.expected)
			}
		})
	}
}

func TestRunShellCommand(t *testing.T) {
	tests := []struct {
		name    string
		command string
		opts    *ExecOptions
		want    string
		wantErr bool
	}{
		{
			name:    "simple_shell_command",
			command: "echo hello",
			opts:    nil,
			want:    "hello\n",
			wantErr: false,
		},
		{
			name:    "shell_command_with_pipe",
			command: "echo hello | tr 'h' 'H'",
			opts:    nil,
			want:    "Hello\n",
			wantErr: false,
		},
		{
			name:    "shell_command_with_multiple_commands",
			command: "echo first; echo second",
			opts:    nil,
			want:    "first\nsecond\n",
			wantErr: false,
		},
		{
			name:    "invalid_shell_command",
			command: "invalidcommand",
			opts:    nil,
			want:    "",
			wantErr: true,
		},
		{
			name:    "shell_command_with_env",
			command: "echo $TEST_SHELL_VAR",
			opts: &ExecOptions{
				Env: []string{"TEST_SHELL_VAR=shell_value"},
			},
			want:    "shell_value\n",
			wantErr: false,
		},
		{
			name:    "shell_command_exit_error",
			command: "exit 1",
			opts:    nil,
			want:    "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			got, err := RunShellCommand(ctx, tt.command, tt.opts)

			if (err != nil) != tt.wantErr {
				t.Errorf("RunShellCommand() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			got = strings.TrimSpace(got)
			want := strings.TrimSpace(tt.want)

			if got != want {
				t.Errorf("RunShellCommand() = %v, want %v", got, want)
			}
		})
	}
}

func TestRunShellCommand_Timeout(t *testing.T) {
	ctx := context.Background()
	opts := &ExecOptions{
		Timeout: 100 * time.Millisecond,
	}

	_, err := RunShellCommand(ctx, "sleep 2", opts)
	if err == nil {
		t.Error("RunShellCommand() expected timeout error")
	}
}
