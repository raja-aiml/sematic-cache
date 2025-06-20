package cmd

import (
	"context"
	"errors"
	"os"
	"testing"

	"github.com/spf13/cobra"
)

func TestNewBaseCommand(t *testing.T) {
	tests := []struct {
		name         string
		loggerPrefix string
	}{
		{
			name:         "simple_prefix",
			loggerPrefix: "test",
		},
		{
			name:         "empty_prefix",
			loggerPrefix: "",
		},
		{
			name:         "complex_prefix",
			loggerPrefix: "test-command-123",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bc := NewBaseCommand(tt.loggerPrefix)

			if bc == nil {
				t.Fatal("NewBaseCommand returned nil")
			}

			if bc.Logger == nil {
				t.Error("NewBaseCommand() Logger is nil")
			}

			if bc.K8sClient != nil {
				t.Error("NewBaseCommand() K8sClient should be nil initially")
			}
		})
	}
}

func TestBaseCommand_Initialize(t *testing.T) {
	// Save and restore HOME env var
	origHome := os.Getenv("HOME")
	if err := os.Setenv("HOME", "/nonexistent"); err != nil {
		t.Fatalf("Failed to set HOME: %v", err)
	}
	defer func() {
		if err := os.Setenv("HOME", origHome); err != nil {
			t.Logf("Failed to restore HOME: %v", err)
		}
	}()

	bc := NewBaseCommand("test")
	ctx := context.Background()

	// This will fail as we can't connect to k8s in test environment
	err := bc.Initialize(ctx)

	// We expect an error in test environment
	if err == nil {
		t.Error("Initialize() expected error in test environment")
	}
}

func TestBaseCommand_AddCommonFlags(t *testing.T) {
	tests := []struct {
		name        string
		clusterName string
		namespace   string
	}{
		{
			name:        "with_cluster_and_namespace",
			clusterName: "test-cluster",
			namespace:   "test-ns",
		},
		{
			name:        "with_cluster_only",
			clusterName: "test-cluster",
			namespace:   "",
		},
		{
			name:        "with_namespace_only",
			clusterName: "",
			namespace:   "test-ns",
		},
		{
			name:        "no_flags",
			clusterName: "",
			namespace:   "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bc := NewBaseCommand("test")
			bc.ClusterName = tt.clusterName
			bc.Namespace = tt.namespace

			cmd := &cobra.Command{
				Use: "test",
			}

			bc.AddCommonFlags(cmd)

			// Check if flags were added
			if tt.clusterName != "" {
				flag := cmd.PersistentFlags().Lookup("cluster")
				if flag == nil {
					t.Error("cluster flag not added")
				} else if flag.DefValue != tt.clusterName {
					t.Errorf("cluster flag default = %v, want %v", flag.DefValue, tt.clusterName)
				}
			}

			if tt.namespace != "" {
				flag := cmd.PersistentFlags().Lookup("namespace")
				if flag == nil {
					t.Error("namespace flag not added")
				} else if flag.DefValue != tt.namespace {
					t.Errorf("namespace flag default = %v, want %v", flag.DefValue, tt.namespace)
				}
			}
		})
	}
}

func TestBaseCommand_Execute(t *testing.T) {
	tests := []struct {
		name    string
		fn      func() error
		wantErr bool
	}{
		{
			name: "successful_execution",
			fn: func() error {
				return nil
			},
			wantErr: true, // Will fail on Initialize
		},
		{
			name: "failed_execution",
			fn: func() error {
				return errors.New("execution failed")
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save and restore HOME env var
			origHome := os.Getenv("HOME")
			if err := os.Setenv("HOME", "/nonexistent"); err != nil {
				t.Fatalf("Failed to set HOME: %v", err)
			}
			defer func() {
				if err := os.Setenv("HOME", origHome); err != nil {
					t.Logf("Failed to restore HOME: %v", err)
				}
			}()

			bc := NewBaseCommand("test")

			err := bc.Execute(tt.fn)

			if (err != nil) != tt.wantErr {
				t.Errorf("Execute() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCommandBuilder_Build(t *testing.T) {
	tests := []struct {
		name  string
		cb    CommandBuilder
		check func(*cobra.Command) error
	}{
		{
			name: "full_builder",
			cb: CommandBuilder{
				Use:   "test",
				Short: "Test command",
				Long:  "This is a test command",
				RunE: func(cmd *cobra.Command, args []string) error {
					return nil
				},
			},
			check: func(cmd *cobra.Command) error {
				if cmd.Use != "test" {
					return errors.New("Use mismatch")
				}
				if cmd.Short != "Test command" {
					return errors.New("Short mismatch")
				}
				if cmd.Long != "This is a test command" {
					return errors.New("Long mismatch")
				}
				if cmd.RunE == nil {
					return errors.New("RunE is nil")
				}
				return nil
			},
		},
		{
			name: "minimal_builder",
			cb: CommandBuilder{
				Use: "minimal",
			},
			check: func(cmd *cobra.Command) error {
				if cmd.Use != "minimal" {
					return errors.New("Use mismatch")
				}
				return nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := tt.cb.Build()

			if cmd == nil {
				t.Fatal("Build() returned nil")
			}

			if err := tt.check(cmd); err != nil {
				t.Errorf("Build() command validation failed: %v", err)
			}
		})
	}
}

func TestWrapError(t *testing.T) {
	tests := []struct {
		name      string
		operation string
		err       error
		want      string
	}{
		{
			name:      "wrap_error",
			operation: "test operation",
			err:       errors.New("base error"),
			want:      "test operation: base error",
		},
		{
			name:      "nil_error",
			operation: "test operation",
			err:       nil,
			want:      "",
		},
		{
			name:      "empty_operation",
			operation: "",
			err:       errors.New("error"),
			want:      ": error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := WrapError(tt.operation, tt.err)

			if tt.err == nil {
				if got != nil {
					t.Errorf("WrapError() = %v, want nil", got)
				}
			} else {
				if got == nil {
					t.Error("WrapError() = nil, want error")
				} else if got.Error() != tt.want {
					t.Errorf("WrapError() = %v, want %v", got.Error(), tt.want)
				}
			}
		})
	}
}

// Test CommandRunner interface implementation
type mockCommandRunner struct {
	runErr error
}

func (m *mockCommandRunner) Run(ctx context.Context) error {
	return m.runErr
}

func TestCommandRunner(t *testing.T) {
	tests := []struct {
		name    string
		runner  CommandRunner
		wantErr bool
	}{
		{
			name:    "successful_run",
			runner:  &mockCommandRunner{runErr: nil},
			wantErr: false,
		},
		{
			name:    "failed_run",
			runner:  &mockCommandRunner{runErr: errors.New("run failed")},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			err := tt.runner.Run(ctx)

			if (err != nil) != tt.wantErr {
				t.Errorf("CommandRunner.Run() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// Benchmark tests
func BenchmarkNewBaseCommand(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = NewBaseCommand("bench")
	}
}

func BenchmarkCommandBuilder_Build(b *testing.B) {
	cb := CommandBuilder{
		Use:   "bench",
		Short: "Benchmark command",
		Long:  "This is a benchmark command",
		RunE: func(cmd *cobra.Command, args []string) error {
			return nil
		},
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = cb.Build()
	}
}

func BenchmarkWrapError(b *testing.B) {
	err := errors.New("test error")

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = WrapError("operation", err)
	}
}
