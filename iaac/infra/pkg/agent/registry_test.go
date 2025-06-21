package agent

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewRegistryBuilder(t *testing.T) {
	builder := NewRegistryBuilder()
	assert.NotNil(t, builder)
	assert.NotNil(t, builder.registry)
	assert.NotNil(t, builder.registry.Commands)
	assert.NotNil(t, builder.registry.Metadata)
}

func TestBuildFromCobraCommand(t *testing.T) {
	// Create test cobra command structure
	rootCmd := &cobra.Command{
		Use:   "test",
		Short: "Test root command",
	}

	clusterCmd := &cobra.Command{
		Use:   "cluster",
		Short: "Manage clusters",
	}

	createCmd := &cobra.Command{
		Use:     "create [name]",
		Short:   "Create a new cluster",
		Example: "test cluster create dev\ntest cluster create prod --nodes=3",
	}
	createCmd.Flags().String("nodes", "1", "Number of nodes")
	createCmd.Flags().Bool("ha", false, "Enable high availability")

	deleteCmd := &cobra.Command{
		Use:   "delete [name]",
		Short: "Delete a cluster",
	}

	clusterCmd.AddCommand(createCmd, deleteCmd)
	rootCmd.AddCommand(clusterCmd)

	// Build registry
	builder := NewRegistryBuilder()
	registry := builder.BuildFromCobraCommand(rootCmd)

	// Verify registry
	assert.NotNil(t, registry)
	assert.Equal(t, "test", registry.Metadata["root_command"])
	assert.Len(t, registry.Commands, 1)

	// Check root command
	rootCommand := registry.Commands[0]
	assert.Equal(t, "test", rootCommand.Name)
	assert.Len(t, rootCommand.Subcommands, 1)

	// Check cluster command
	clusterCommand := rootCommand.Subcommands[0]
	assert.Equal(t, "cluster", clusterCommand.Name)
	assert.Len(t, clusterCommand.Subcommands, 2)

	// Check create command
	createCommand := clusterCommand.Subcommands[0]
	assert.Equal(t, "create", createCommand.Name)
	assert.Len(t, createCommand.Options, 2)
	assert.Len(t, createCommand.Examples, 2)

	// Check delete command
	deleteCommand := clusterCommand.Subcommands[1]
	assert.Equal(t, "delete", deleteCommand.Name)
	assert.True(t, deleteCommand.Dangerous)
}

func TestIsDangerousCommand(t *testing.T) {
	builder := NewRegistryBuilder()

	tests := []struct {
		name     string
		cmdName  string
		expected bool
	}{
		{"delete command", "delete", true},
		{"remove command", "remove", true},
		{"destroy command", "destroy", true},
		{"reset command", "reset", true},
		{"purge command", "purge", true},
		{"create command", "create", false},
		{"list command", "list", false},
		{"get command", "get", false},
		{"mixed case DELETE", "DELETE", true},
		{"contains delete", "delete-all", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := builder.isDangerousCommand(tt.cmdName)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestRequiresConfirmation(t *testing.T) {
	builder := NewRegistryBuilder()

	tests := []struct {
		name     string
		cmdName  string
		expected bool
	}{
		{"delete command", "delete", true},
		{"create command", "create", true},
		{"update command", "update", true},
		{"apply command", "apply", true},
		{"deploy command", "deploy", true},
		{"list command", "list", false},
		{"get command", "get", false},
		{"show command", "show", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := builder.requiresConfirmation(tt.cmdName)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestSaveAndLoadRegistry(t *testing.T) {
	// Create temporary directory
	tmpDir, err := os.MkdirTemp("", "registry-test")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	// Create test registry
	builder := NewRegistryBuilder()
	builder.registry = CreateDefaultRegistry()

	// Save to file
	registryPath := filepath.Join(tmpDir, "test-registry.json")
	err = builder.SaveToFile(registryPath)
	require.NoError(t, err)

	// Load from file
	loaded, err := LoadRegistryFromFile(registryPath)
	require.NoError(t, err)

	// Verify loaded registry
	assert.Equal(t, builder.registry.Version, loaded.Version)
	assert.Len(t, loaded.Commands, len(builder.registry.Commands))
	assert.Equal(t, builder.registry.Commands[0].Name, loaded.Commands[0].Name)
}

func TestGenerateMarkdownDoc(t *testing.T) {
	builder := NewRegistryBuilder()
	builder.registry = CreateDefaultRegistry()

	markdown := builder.GenerateMarkdownDoc()

	// Verify markdown content
	assert.Contains(t, markdown, "# Command Reference")
	assert.Contains(t, markdown, "## cluster")
	assert.Contains(t, markdown, "### create")
	assert.Contains(t, markdown, "⚠️ **Warning**")
	assert.Contains(t, markdown, "### Options")
	assert.Contains(t, markdown, "### Examples")
}

func TestFindCommand(t *testing.T) {
	registry := CreateDefaultRegistry()

	tests := []struct {
		name        string
		path        []string
		shouldFind  bool
		commandName string
	}{
		{"find root command", []string{"cluster"}, true, "cluster"},
		{"find subcommand", []string{"cluster", "create"}, true, "create"},
		{"find nested subcommand", []string{"blueprint", "validate"}, true, "validate"},
		{"empty path", []string{}, false, ""},
		{"invalid command", []string{"invalid"}, false, ""},
		{"invalid subcommand", []string{"cluster", "invalid"}, false, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd, err := registry.FindCommand(tt.path)

			if tt.shouldFind {
				require.NoError(t, err)
				assert.NotNil(t, cmd)
				assert.Equal(t, tt.commandName, cmd.Name)
			} else {
				assert.Error(t, err)
				assert.Nil(t, cmd)
			}
		})
	}
}

func TestGetCommandPath(t *testing.T) {
	tests := []struct {
		name      string
		cmd       Command
		ancestors []string
		expected  string
	}{
		{
			name:      "root command",
			cmd:       Command{Name: "cluster"},
			ancestors: []string{},
			expected:  "cluster",
		},
		{
			name:      "subcommand",
			cmd:       Command{Name: "create"},
			ancestors: []string{"cluster"},
			expected:  "cluster create",
		},
		{
			name:      "nested subcommand",
			cmd:       Command{Name: "validate"},
			ancestors: []string{"blueprint", "schema"},
			expected:  "blueprint schema validate",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetCommandPath(tt.cmd, tt.ancestors...)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestCreateDefaultRegistry(t *testing.T) {
	registry := CreateDefaultRegistry()

	assert.NotNil(t, registry)
	assert.Equal(t, "1.0.0", registry.Version)
	assert.NotNil(t, registry.GeneratedAt)
	assert.Greater(t, len(registry.Commands), 0)

	// Verify cluster command exists
	clusterCmd, err := registry.FindCommand([]string{"cluster"})
	require.NoError(t, err)
	assert.Equal(t, "cluster", clusterCmd.Name)
	assert.Greater(t, len(clusterCmd.Subcommands), 0)

	// Verify dangerous command marking
	deleteCmd, err := registry.FindCommand([]string{"cluster", "delete"})
	require.NoError(t, err)
	assert.True(t, deleteCmd.Dangerous)
	assert.True(t, deleteCmd.RequiresConfirmation)
}

func TestProcessCobraCommandWithHiddenCommands(t *testing.T) {
	rootCmd := &cobra.Command{
		Use:   "test",
		Short: "Test command",
	}

	visibleCmd := &cobra.Command{
		Use:   "visible",
		Short: "Visible command",
	}

	hiddenCmd := &cobra.Command{
		Use:    "hidden",
		Short:  "Hidden command",
		Hidden: true,
	}

	rootCmd.AddCommand(visibleCmd, hiddenCmd)

	builder := NewRegistryBuilder()
	registry := builder.BuildFromCobraCommand(rootCmd)

	// Verify only visible command is included
	assert.Len(t, registry.Commands[0].Subcommands, 1)
	assert.Equal(t, "visible", registry.Commands[0].Subcommands[0].Name)
}

func TestCommandOptionsExtraction(t *testing.T) {
	cmd := &cobra.Command{
		Use:   "test",
		Short: "Test command",
	}

	// Add various flag types
	cmd.Flags().String("string-flag", "default", "A string flag")
	cmd.Flags().StringP("short-flag", "s", "", "Flag with shorthand")
	cmd.Flags().Int("int-flag", 42, "An integer flag")
	cmd.Flags().Bool("bool-flag", false, "A boolean flag")

	builder := NewRegistryBuilder()
	command := builder.processCobraCommand(cmd, "")

	assert.Len(t, command.Options, 4)

	// Find string flag
	var stringOpt *Option
	for _, opt := range command.Options {
		if opt.Name == "string-flag" {
			stringOpt = &opt
			break
		}
	}

	require.NotNil(t, stringOpt)
	assert.Equal(t, "string", stringOpt.Type)
	assert.Equal(t, "default", stringOpt.Default)
	assert.Equal(t, "A string flag", stringOpt.Description)

	// Find short flag
	var shortOpt *Option
	for _, opt := range command.Options {
		if opt.Name == "short-flag" {
			shortOpt = &opt
			break
		}
	}

	require.NotNil(t, shortOpt)
	assert.Equal(t, "s", shortOpt.Shorthand)
}

func TestMarkdownDocGeneration(t *testing.T) {
	builder := NewRegistryBuilder()

	// Create a simple command for testing
	builder.registry.Commands = []Command{
		{
			Name:                 "test",
			Description:          "Test command",
			Dangerous:            true,
			RequiresConfirmation: true,
			Options: []Option{
				{
					Name:        "option1",
					Shorthand:   "o",
					Type:        "string",
					Description: "Test option",
					Required:    true,
					Default:     "default",
				},
			},
			Examples: []Example{
				{
					Command:     "test --option1=value",
					Description: "Example usage",
				},
			},
			Subcommands: []Command{
				{
					Name:        "subcommand",
					Description: "Test subcommand",
				},
			},
		},
	}

	markdown := builder.GenerateMarkdownDoc()

	// Verify all sections are present
	assert.Contains(t, markdown, "# Command Reference")
	assert.Contains(t, markdown, "## test")
	assert.Contains(t, markdown, "Test command")
	assert.Contains(t, markdown, "⚠️ **Warning**")
	assert.Contains(t, markdown, "ℹ️ **Note**")
	assert.Contains(t, markdown, "### Options")
	assert.Contains(t, markdown, "--option1, -o")
	assert.Contains(t, markdown, "### Examples")
	assert.Contains(t, markdown, "```bash")
	assert.Contains(t, markdown, "test --option1=value")
	assert.Contains(t, markdown, "### Subcommands")
	assert.Contains(t, markdown, "#### subcommand")
}

func TestCommandExampleParsing(t *testing.T) {
	cmd := &cobra.Command{
		Use:   "test",
		Short: "Test command",
		Example: `test create foo
test create bar --option=value
test delete baz`,
	}

	builder := NewRegistryBuilder()
	command := builder.processCobraCommand(cmd, "")

	assert.Len(t, command.Examples, 3)

	// Verify examples are parsed correctly
	examples := []string{
		"test create foo",
		"test create bar --option=value",
		"test delete baz",
	}

	for i, example := range command.Examples {
		assert.Equal(t, examples[i], example.Command)
		assert.Contains(t, example.Description, "Example usage")
	}
}
