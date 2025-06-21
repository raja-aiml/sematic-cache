package agent

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

// RegistryBuilder builds command documentation from cobra commands
type RegistryBuilder struct {
	registry *CommandRegistry
}

// NewRegistryBuilder creates a new registry builder
func NewRegistryBuilder() *RegistryBuilder {
	return &RegistryBuilder{
		registry: &CommandRegistry{
			Commands:    []Command{},
			Version:     "1.0.0",
			GeneratedAt: time.Now(),
			Metadata:    make(map[string]string),
		},
	}
}

// BuildFromCobraCommand builds registry from cobra root command
func (rb *RegistryBuilder) BuildFromCobraCommand(rootCmd *cobra.Command) *CommandRegistry {
	rb.registry.Metadata["root_command"] = rootCmd.Use
	rb.registry.Metadata["description"] = rootCmd.Short

	// Process root command
	rootCommand := rb.processCobraCommand(rootCmd, "")
	rb.registry.Commands = append(rb.registry.Commands, rootCommand)

	return rb.registry
}

// processCobraCommand recursively processes cobra commands
func (rb *RegistryBuilder) processCobraCommand(cmd *cobra.Command, category string) Command {
	command := Command{
		Name:                 cmd.Use,
		Description:          cmd.Short,
		Category:             category,
		Subcommands:          []Command{},
		Options:              []Option{},
		Examples:             []Example{},
		Dangerous:            rb.isDangerousCommand(cmd.Use),
		RequiresConfirmation: rb.requiresConfirmation(cmd.Use),
	}

	// Extract command name without arguments
	parts := strings.Fields(cmd.Use)
	if len(parts) > 0 {
		command.Name = parts[0]
	}

	// Process flags
	cmd.Flags().VisitAll(func(flag *pflag.Flag) {
		option := Option{
			Name:        flag.Name,
			Shorthand:   flag.Shorthand,
			Type:        flag.Value.Type(),
			Description: flag.Usage,
			Default:     flag.DefValue,
		}

		// Check if required
		if cmd.Flags().Lookup(flag.Name) != nil {
			required := cmd.Flags().Lookup(flag.Name)
			if required != nil && required.Annotations != nil {
				if _, ok := required.Annotations[cobra.BashCompOneRequiredFlag]; ok {
					option.Required = true
				}
			}
		}

		command.Options = append(command.Options, option)
	})

	// Process examples
	if cmd.Example != "" {
		examples := strings.Split(cmd.Example, "\n")
		for _, ex := range examples {
			ex = strings.TrimSpace(ex)
			if ex != "" {
				command.Examples = append(command.Examples, Example{
					Command:     ex,
					Description: fmt.Sprintf("Example usage of %s", cmd.Name()),
				})
			}
		}
	}

	// Process subcommands
	for _, subCmd := range cmd.Commands() {
		if !subCmd.Hidden {
			subCategory := category
			if subCategory == "" {
				subCategory = command.Name
			} else {
				subCategory = fmt.Sprintf("%s/%s", subCategory, command.Name)
			}
			subCommand := rb.processCobraCommand(subCmd, subCategory)
			command.Subcommands = append(command.Subcommands, subCommand)
		}
	}

	return command
}

// isDangerousCommand checks if a command is potentially dangerous
func (rb *RegistryBuilder) isDangerousCommand(cmdName string) bool {
	dangerousPatterns := []string{
		"delete", "remove", "destroy", "reset", "purge",
		"drop", "truncate", "wipe", "clear",
	}

	cmdLower := strings.ToLower(cmdName)
	for _, pattern := range dangerousPatterns {
		if strings.Contains(cmdLower, pattern) {
			return true
		}
	}

	return false
}

// requiresConfirmation checks if a command requires user confirmation
func (rb *RegistryBuilder) requiresConfirmation(cmdName string) bool {
	confirmPatterns := []string{
		"delete", "remove", "destroy", "apply", "create",
		"update", "modify", "deploy", "rollback",
	}

	cmdLower := strings.ToLower(cmdName)
	for _, pattern := range confirmPatterns {
		if strings.Contains(cmdLower, pattern) {
			return true
		}
	}

	return false
}

// SaveToFile saves the registry to a JSON file
func (rb *RegistryBuilder) SaveToFile(filepath string) error {
	data, err := json.MarshalIndent(rb.registry, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal registry: %w", err)
	}

	if err := os.WriteFile(filepath, data, 0644); err != nil {
		return fmt.Errorf("failed to write registry file: %w", err)
	}

	return nil
}

// LoadFromFile loads a registry from a JSON file
func LoadRegistryFromFile(filepath string) (*CommandRegistry, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read registry file: %w", err)
	}

	var registry CommandRegistry
	if err := json.Unmarshal(data, &registry); err != nil {
		return nil, fmt.Errorf("failed to unmarshal registry: %w", err)
	}

	return &registry, nil
}

// GenerateMarkdownDoc generates markdown documentation from the registry
func (rb *RegistryBuilder) GenerateMarkdownDoc() string {
	var sb strings.Builder

	sb.WriteString("# Command Reference\n\n")
	sb.WriteString(fmt.Sprintf("Generated at: %s\n\n", rb.registry.GeneratedAt.Format(time.RFC3339)))
	sb.WriteString(fmt.Sprintf("Version: %s\n\n", rb.registry.Version))

	for _, cmd := range rb.registry.Commands {
		rb.writeCommandDoc(&sb, cmd, 2)
	}

	return sb.String()
}

// writeCommandDoc writes documentation for a single command
func (rb *RegistryBuilder) writeCommandDoc(sb *strings.Builder, cmd Command, level int) {
	prefix := strings.Repeat("#", level)

	sb.WriteString(fmt.Sprintf("%s %s\n\n", prefix, cmd.Name))
	sb.WriteString(fmt.Sprintf("%s\n\n", cmd.Description))

	if cmd.Dangerous {
		sb.WriteString("⚠️ **Warning**: This is a dangerous command that can cause data loss.\n\n")
	}

	if cmd.RequiresConfirmation {
		sb.WriteString("ℹ️ **Note**: This command requires confirmation before execution.\n\n")
	}

	// Write options
	if len(cmd.Options) > 0 {
		sb.WriteString(fmt.Sprintf("%s Options\n\n", prefix+"#"))
		sb.WriteString("| Option | Type | Description | Required | Default |\n")
		sb.WriteString("|--------|------|-------------|----------|----------|\n")

		for _, opt := range cmd.Options {
			required := "No"
			if opt.Required {
				required = "Yes"
			}

			shorthand := ""
			if opt.Shorthand != "" {
				shorthand = fmt.Sprintf(", -%s", opt.Shorthand)
			}

			sb.WriteString(fmt.Sprintf("| --%s%s | %s | %s | %s | %s |\n",
				opt.Name, shorthand, opt.Type, opt.Description, required, opt.Default))
		}
		sb.WriteString("\n")
	}

	// Write examples
	if len(cmd.Examples) > 0 {
		sb.WriteString(fmt.Sprintf("%s Examples\n\n", prefix+"#"))
		for _, ex := range cmd.Examples {
			sb.WriteString(fmt.Sprintf("```bash\n%s\n```\n", ex.Command))
			if ex.Description != "" {
				sb.WriteString(fmt.Sprintf("%s\n", ex.Description))
			}
			sb.WriteString("\n")
		}
	}

	// Write subcommands
	if len(cmd.Subcommands) > 0 {
		sb.WriteString(fmt.Sprintf("%s Subcommands\n\n", prefix+"#"))
		for _, subCmd := range cmd.Subcommands {
			rb.writeCommandDoc(sb, subCmd, level+2)
		}
	}
}

// CreateDefaultRegistry creates a registry with default iaac commands
func CreateDefaultRegistry() *CommandRegistry {
	return &CommandRegistry{
		Version:     "1.0.0",
		GeneratedAt: time.Now(),
		Commands: []Command{
			{
				Name:        "cluster",
				Description: "Manage k3d clusters",
				Category:    "infrastructure",
				Subcommands: []Command{
					{
						Name:        "create",
						Description: "Create a new k3d cluster",
						Options: []Option{
							{Name: "name", Type: "string", Description: "Cluster name", Required: true},
							{Name: "nodes", Type: "int", Description: "Number of nodes", Default: "1"},
							{Name: "k3s-version", Type: "string", Description: "K3s version to use"},
						},
						Examples: []Example{
							{Command: "iaac cluster create --name=dev --nodes=3", Description: "Create a 3-node cluster"},
						},
					},
					{
						Name:                 "delete",
						Description:          "Delete a k3d cluster",
						Dangerous:            true,
						RequiresConfirmation: true,
						Options: []Option{
							{Name: "name", Type: "string", Description: "Cluster name", Required: true},
							{Name: "force", Type: "bool", Description: "Force deletion"},
						},
					},
					{
						Name:        "list",
						Description: "List all k3d clusters",
						Options: []Option{
							{Name: "output", Shorthand: "o", Type: "string", Description: "Output format", Choices: []string{"json", "yaml", "table"}},
						},
					},
				},
			},
			{
				Name:        "deploy",
				Description: "Deploy applications to cluster",
				Category:    "deployment",
				Subcommands: []Command{
					{
						Name:                 "apply",
						Description:          "Apply kubernetes manifests",
						RequiresConfirmation: true,
						Options: []Option{
							{Name: "file", Shorthand: "f", Type: "string", Description: "Manifest file path", Required: true},
							{Name: "namespace", Shorthand: "n", Type: "string", Description: "Target namespace", Default: "default"},
							{Name: "dry-run", Type: "bool", Description: "Perform a dry run"},
						},
					},
				},
			},
			{
				Name:        "blueprint",
				Description: "Manage infrastructure blueprints",
				Category:    "configuration",
				Subcommands: []Command{
					{
						Name:        "list",
						Description: "List available blueprints",
						Options: []Option{
							{Name: "path", Type: "string", Description: "Blueprint directory path"},
						},
					},
					{
						Name:        "validate",
						Description: "Validate a blueprint",
						Options: []Option{
							{Name: "file", Shorthand: "f", Type: "string", Description: "Blueprint file", Required: true},
						},
					},
				},
			},
		},
		Metadata: map[string]string{
			"tool":        "iaac",
			"description": "Infrastructure as Code management tool",
		},
	}
}

// FindCommand searches for a command in the registry
func (cr *CommandRegistry) FindCommand(path []string) (*Command, error) {
	if len(path) == 0 {
		return nil, fmt.Errorf("empty command path")
	}

	commands := cr.Commands
	var currentCmd *Command

	for i, part := range path {
		found := false
		for _, cmd := range commands {
			if cmd.Name == part {
				currentCmd = &cmd
				commands = cmd.Subcommands
				found = true
				break
			}
		}

		if !found {
			return nil, fmt.Errorf("command not found: %s", strings.Join(path[:i+1], " "))
		}
	}

	return currentCmd, nil
}

// GetCommandPath returns the full command path as a string
func GetCommandPath(cmd Command, ancestors ...string) string {
	path := append(ancestors, cmd.Name)
	return strings.Join(path, " ")
}
