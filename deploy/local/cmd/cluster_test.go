package cmd

import (
	"bytes"
	"strings"
	"testing"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/constants"
)

func TestClusterCmd(t *testing.T) {
	cmd := ClusterCmd()

	if cmd == nil {
		t.Fatal("ClusterCmd() returned nil")
	}

	if cmd.Use != "cluster" {
		t.Errorf("ClusterCmd().Use = %v, want %v", cmd.Use, "cluster")
	}

	// Check that subcommands are added
	subcommands := cmd.Commands()
	if len(subcommands) == 0 {
		t.Error("ClusterCmd() has no subcommands")
	}

	// Check specific commands exist
	commandNames := make(map[string]bool)
	for _, subcmd := range subcommands {
		commandNames[subcmd.Use] = true
	}

	expectedCommands := []string{"up", "down", "ps", "logs", "test"}
	for _, expected := range expectedCommands {
		if !commandNames[expected] {
			t.Errorf("ClusterCmd() missing command %q", expected)
		}
	}

	// Check flags
	flag := cmd.PersistentFlags().Lookup("name")
	if flag == nil {
		t.Error("ClusterCmd() missing 'name' flag")
	} else if flag.DefValue != constants.DefaultClusterName {
		t.Errorf("ClusterCmd() name flag default = %v, want %v", flag.DefValue, constants.DefaultClusterName)
	}
}

func TestClusterUpCmd(t *testing.T) {
	cmd := clusterUpCmd("test-cluster")

	if cmd == nil {
		t.Fatal("clusterUpCmd() returned nil")
	}

	if cmd.Use != "up" {
		t.Errorf("clusterUpCmd().Use = %v, want %v", cmd.Use, "up")
	}

	if cmd.RunE == nil {
		t.Error("clusterUpCmd().RunE is nil")
	}
}

func TestClusterDownCmd(t *testing.T) {
	cmd := clusterDownCmd("test-cluster")

	if cmd == nil {
		t.Fatal("clusterDownCmd() returned nil")
	}

	if cmd.Use != "down" {
		t.Errorf("clusterDownCmd().Use = %v, want %v", cmd.Use, "down")
	}

	if cmd.RunE == nil {
		t.Error("clusterDownCmd().RunE is nil")
	}
}

func TestClusterStatusCmd(t *testing.T) {
	cmd := clusterStatusCmd("test-cluster")

	if cmd == nil {
		t.Fatal("clusterStatusCmd() returned nil")
	}

	if cmd.Use != "ps" {
		t.Errorf("clusterStatusCmd().Use = %v, want %v", cmd.Use, "ps")
	}

	if cmd.RunE == nil {
		t.Error("clusterStatusCmd().RunE is nil")
	}
}

func TestClusterLogsCmd(t *testing.T) {
	cmd := clusterLogsCmd("test-cluster")

	if cmd == nil {
		t.Fatal("clusterLogsCmd() returned nil")
	}

	if cmd.Use != "logs" {
		t.Errorf("clusterLogsCmd().Use = %v, want %v", cmd.Use, "logs")
	}

	if cmd.RunE == nil {
		t.Error("clusterLogsCmd().RunE is nil")
	}

	// Check flags
	flags := []struct {
		name     string
		defValue string
	}{
		{"namespace", constants.AppNamespace},
		{"selector", ""},
		{"tail", "50"},
	}

	for _, f := range flags {
		flag := cmd.Flags().Lookup(f.name)
		if flag == nil {
			t.Errorf("clusterLogsCmd() missing '%s' flag", f.name)
		} else if flag.DefValue != f.defValue {
			t.Errorf("clusterLogsCmd() %s flag default = %v, want %v", f.name, flag.DefValue, f.defValue)
		}
	}
}

func TestClusterTestCmd(t *testing.T) {
	cmd := clusterTestCmd("test-cluster")

	if cmd == nil {
		t.Fatal("clusterTestCmd() returned nil")
	}

	if cmd.Use != "test" {
		t.Errorf("clusterTestCmd().Use = %v, want %v", cmd.Use, "test")
	}

	if cmd.RunE == nil {
		t.Error("clusterTestCmd().RunE is nil")
	}
}

func TestWaitForInfrastructure(t *testing.T) {
	// Skip this test as it requires a real k8s client and cluster
	t.Skip("Skipping TestWaitForInfrastructure - requires real k8s cluster")
}

// Helper function to execute command
func executeClusterCommand(args ...string) (output string, err error) {
	cmd := ClusterCmd()
	buf := new(bytes.Buffer)
	cmd.SetOut(buf)
	cmd.SetErr(buf)
	cmd.SetArgs(args)

	err = cmd.Execute()
	return buf.String(), err
}

func TestClusterCmdExecution(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		wantErr bool
	}{
		{
			name:    "no_args",
			args:    []string{},
			wantErr: false, // Shows help
		},
		{
			name:    "help",
			args:    []string{"--help"},
			wantErr: false,
		},
		{
			name:    "invalid_subcommand",
			args:    []string{"invalid"},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := executeClusterCommand(tt.args...)

			if (err != nil) != tt.wantErr {
				t.Errorf("executeClusterCommand() error = %v, wantErr %v", err, tt.wantErr)
			}

			if !tt.wantErr && !strings.Contains(output, "Create, destroy, and manage k3d clusters") {
				t.Errorf("executeClusterCommand() output missing expected help text. Got: %q", output)
			}
		})
	}
}

// Benchmark tests
func BenchmarkClusterCmd(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = ClusterCmd()
	}
}

func BenchmarkClusterUpCmd(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = clusterUpCmd("bench-cluster")
	}
}
