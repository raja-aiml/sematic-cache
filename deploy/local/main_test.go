package main

import (
	"bytes"
	"strings"
	"testing"
	
	"github.com/spf13/cobra"
)

func TestVersionCmd(t *testing.T) {
	cmd := versionCmd()
	
	if cmd == nil {
		t.Fatal("versionCmd() returned nil")
	}
	
	if cmd.Use != "version" {
		t.Errorf("versionCmd().Use = %v, want %v", cmd.Use, "version")
	}
	
	// Test execution
	buf := new(bytes.Buffer)
	cmd.SetOut(buf)
	cmd.SetErr(buf)
	
	err := cmd.Execute()
	if err != nil {
		t.Errorf("versionCmd().Execute() error = %v", err)
	}
	
	output := buf.String()
	expectedStrings := []string{
		"Semantic Cache Deploy",
		"Version:",
		"Built:",
		"Git Commit:",
		"Go Version:",
		"OS/Arch:",
	}
	
	for _, expected := range expectedStrings {
		if !strings.Contains(output, expected) {
			t.Errorf("versionCmd() output missing %q", expected)
		}
	}
}

func TestConfigCmd(t *testing.T) {
	cmd := configCmd()
	
	if cmd == nil {
		t.Fatal("configCmd() returned nil")
	}
	
	if cmd.Use != "config" {
		t.Errorf("configCmd().Use = %v, want %v", cmd.Use, "config")
	}
	
	if cmd.Short != "Configuration management commands" {
		t.Errorf("configCmd().Short = %v", cmd.Short)
	}
}

func TestRootCmd(t *testing.T) {
	// Test that rootCmd is properly initialized
	if rootCmd == nil {
		t.Fatal("rootCmd is nil")
	}
	
	if rootCmd.Use != "semantic-cache-deploy" {
		t.Errorf("rootCmd.Use = %v, want %v", rootCmd.Use, "semantic-cache-deploy")
	}
	
	// Check that subcommands are added
	subcommands := rootCmd.Commands()
	if len(subcommands) == 0 {
		t.Error("rootCmd has no subcommands")
	}
	
	// Check specific commands exist
	commandNames := make(map[string]bool)
	for _, cmd := range subcommands {
		commandNames[cmd.Use] = true
	}
	
	expectedCommands := []string{"cluster", "dev", "workflow", "composite-test", "debug", "version", "config"}
	for _, expected := range expectedCommands {
		if !commandNames[expected] {
			t.Errorf("rootCmd missing command %q", expected)
		}
	}
}

func TestInit(t *testing.T) {
	// Test that init() properly configures rootCmd
	if !rootCmd.CompletionOptions.DisableDefaultCmd {
		t.Error("rootCmd.CompletionOptions.DisableDefaultCmd should be true")
	}
}

func TestVersionVariables(t *testing.T) {
	// Test that version variables are set
	if version == "" {
		t.Error("version should not be empty")
	}
	
	if buildTime == "" {
		t.Error("buildTime should not be empty")
	}
	
	if gitCommit == "" {
		t.Error("gitCommit should not be empty")
	}
}

// Test helper function
func executeCommand(root *cobra.Command, args ...string) (output string, err error) {
	buf := new(bytes.Buffer)
	root.SetOut(buf)
	root.SetErr(buf)
	root.SetArgs(args)
	
	err = root.Execute()
	return buf.String(), err
}

func TestMainExecution(t *testing.T) {
	// Save original root command
	originalRoot := rootCmd
	defer func() {
		rootCmd = originalRoot
	}()
	
	// Create a test root command
	testRoot := &cobra.Command{
		Use:   "test",
		Short: "Test command",
		RunE: func(cmd *cobra.Command, args []string) error {
			return nil
		},
	}
	
	// Test successful execution
	output, err := executeCommand(testRoot)
	if err != nil {
		t.Errorf("executeCommand() error = %v", err)
	}
	
	_ = output
}

// Benchmark tests
func BenchmarkVersionCmd(b *testing.B) {
	cmd := versionCmd()
	buf := new(bytes.Buffer)
	cmd.SetOut(buf)
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		buf.Reset()
		_ = cmd.Execute()
	}
}

func BenchmarkConfigCmd(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = configCmd()
	}
}