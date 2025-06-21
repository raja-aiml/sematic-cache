package cmd

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/fatih/color"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/agent"
	pkgconfig "github.com/raja-aiml/sematic-cache/deploy/local/pkg/config"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	// Agent command flags
	agentInteractive bool
	agentConfigFile  string

	// Docs command flags
	docsOutputFile   string
	docsOutputFormat string
)

// AgentCmd creates the agent command
func AgentCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "agent [query]",
		Short: "Natural language interface for infrastructure management",
		Long: `The agent command provides a natural language interface to execute infrastructure
management commands. You can ask questions or give instructions in plain English.

Examples:
  # Execute a single query
  iaac agent "create a new cluster with 3 nodes"
  
  # Start interactive mode
  iaac agent --interactive
  
  # Use custom configuration
  iaac agent --config agent.yaml "show all clusters"`,
		RunE: runAgent,
	}

	cmd.Flags().BoolVarP(&agentInteractive, "interactive", "i", false, "Start interactive mode")
	cmd.Flags().StringVarP(&agentConfigFile, "config", "c", "", "Agent configuration file")

	// Add configuration flags
	cmd.Flags().String("openai-model", "gpt-4-turbo-preview", "OpenAI model to use")
	cmd.Flags().Int("openai-max-tokens", 1000, "Maximum tokens for OpenAI responses")
	cmd.Flags().Bool("enable-dangerous", false, "Enable execution of dangerous commands")
	cmd.Flags().Bool("require-confirmation", true, "Require confirmation before execution")
	cmd.Flags().Duration("command-timeout", 120*time.Second, "Command execution timeout")
	cmd.Flags().String("audit-log", "audit.log", "Audit log file path")

	// Bind flags to viper
	viper.BindPFlag("openai-model", cmd.Flags().Lookup("openai-model"))
	viper.BindPFlag("openai-max-tokens", cmd.Flags().Lookup("openai-max-tokens"))
	viper.BindPFlag("enable-dangerous", cmd.Flags().Lookup("enable-dangerous"))
	viper.BindPFlag("require-confirmation", cmd.Flags().Lookup("require-confirmation"))
	viper.BindPFlag("command-timeout", cmd.Flags().Lookup("command-timeout"))
	viper.BindPFlag("audit-log", cmd.Flags().Lookup("audit-log"))

	return cmd
}

// DocsCmd creates the docs command
func DocsCmd(rootCmd *cobra.Command, version string) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "docs",
		Short: "Generate command documentation",
		Long: `Generate comprehensive documentation for all available commands.
This documentation can be used by the NLP agent to understand available operations.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runDocs(cmd, args, rootCmd, version)
		},
	}

	cmd.Flags().StringVarP(&docsOutputFile, "output", "o", "commands.json", "Output file path")
	cmd.Flags().StringVarP(&docsOutputFormat, "format", "f", "json", "Output format (json, markdown)")

	return cmd
}

func runAgent(cmd *cobra.Command, args []string) error {
	// Load configuration
	config, err := loadAgentConfig(cmd)
	if err != nil {
		return fmt.Errorf("failed to load configuration: %w", err)
	}

	// Create agent
	cliAgent, err := agent.NewCLIAgent(config)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	ctx := context.Background()

	// Check if interactive mode or query provided
	if agentInteractive {
		return cliAgent.StartInteractive(ctx)
	}

	// Single query mode
	if len(args) == 0 {
		return fmt.Errorf("please provide a query or use --interactive flag")
	}

	query := args[0]

	// Process the query
	fmt.Println(color.YellowString("Processing query..."))
	result, err := cliAgent.ProcessQuery(ctx, query)
	if err != nil {
		return err
	}

	if !result.Success {
		fmt.Println(color.RedString("✗ Failed to process query"))
		fmt.Printf("Error: %s\n", result.Error)

		if len(result.Suggestions) > 0 {
			fmt.Println("\nSuggestions:")
			for _, suggestion := range result.Suggestions {
				fmt.Printf("  • %s\n", suggestion)
			}
		}
		return fmt.Errorf("query processing failed")
	}

	// Display interpreted command
	interpretedCmd := result.Command
	fmt.Println(color.GreenString("\n✓ Query interpreted successfully"))
	fmt.Printf("Command: %s\n", color.CyanString(formatCommand(interpretedCmd)))
	fmt.Printf("Confidence: %.0f%%\n", interpretedCmd.Confidence*100)

	if interpretedCmd.Explanation != "" {
		fmt.Printf("Explanation: %s\n", interpretedCmd.Explanation)
	}

	if interpretedCmd.Dangerous {
		fmt.Println(color.YellowString("\n⚠️  Warning: This is a dangerous command"))
	}

	// Execute if not in dry-run mode
	if !viper.GetBool("dry-run") {
		fmt.Println("\nExecuting command...")
		execResult, err := cliAgent.ExecuteQuery(ctx, query)
		if err != nil {
			return fmt.Errorf("execution failed: %w", err)
		}

		displayExecutionResult(execResult)
	} else {
		fmt.Println("\n(Dry run - command not executed)")
	}

	return nil
}

func loadAgentConfig(cmd *cobra.Command) (*agent.Config, error) {
	// Start with defaults
	config := &agent.Config{
		OpenAIModel:         "gpt-4-turbo-preview",
		OpenAIMaxTokens:     1000,
		RequireConfirmation: true,
		CommandTimeout:      30 * time.Second,
		AuditLogPath:        "audit.log",
		InteractivePrompt:   "iaac> ",
		EnableAutoComplete:  true,
		CommandFactories: map[string]func() *cobra.Command{
			"cluster":  ClusterCmd,
			"dev":      DevCmd,
			"workflow": WorkflowCmd,
			"test":     TestCmd,
			"validate": ValidateCmd,
			"manifest": ManifestCmd,
		},
	}

	// Load from environment
	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		config.OpenAIKey = apiKey
	}

	// Load from config file if specified
	if agentConfigFile != "" {
		viper.SetConfigFile(agentConfigFile)
	} else {
		// Try to find agent.yaml in config directory
		if configPaths, err := pkgconfig.ResolveConfigPaths(cmd); err == nil {
			agentConfigPath := filepath.Join(configPaths.ConfigDir, "agent.yaml")
			if _, err := os.Stat(agentConfigPath); err == nil {
				viper.SetConfigFile(agentConfigPath)
			}
		}
	}

	// Read config if file was set
	if viper.ConfigFileUsed() != "" {
		if err := viper.ReadInConfig(); err != nil {
			// Don't fail if config file doesn't exist
			if !os.IsNotExist(err) {
				return nil, fmt.Errorf("failed to read config file: %w", err)
			}
		} else {
			if err := viper.Unmarshal(config); err != nil {
				return nil, fmt.Errorf("failed to unmarshal config: %w", err)
			}
		}
	}

	// Override with command line flags
	if model := viper.GetString("openai-model"); model != "" {
		config.OpenAIModel = model
	}
	if maxTokens := viper.GetInt("openai-max-tokens"); maxTokens > 0 {
		config.OpenAIMaxTokens = maxTokens
	}
	config.EnableDangerousCommands = viper.GetBool("enable-dangerous")
	config.RequireConfirmation = viper.GetBool("require-confirmation")
	if timeout := viper.GetDuration("command-timeout"); timeout > 0 {
		config.CommandTimeout = timeout
	}
	if auditLog := viper.GetString("audit-log"); auditLog != "" {
		config.AuditLogPath = auditLog
	}

	// Validate configuration
	if config.OpenAIKey == "" {
		return nil, fmt.Errorf("OpenAI API key not set. Please set OPENAI_API_KEY environment variable")
	}

	return config, nil
}

func formatCommand(cmd *agent.InterpretedCommand) string {
	parts := []string{cmd.Command}

	if cmd.Subcommand != "" {
		parts = append(parts, cmd.Subcommand)
	}

	parts = append(parts, cmd.Args...)

	for key, value := range cmd.Options {
		if len(key) == 1 {
			parts = append(parts, fmt.Sprintf("-%s %s", key, value))
		} else {
			parts = append(parts, fmt.Sprintf("--%s %s", key, value))
		}
	}

	return strings.Join(parts, " ")
}

func displayExecutionResult(result *agent.ExecutionResult) {
	fmt.Println("\n" + strings.Repeat("-", 50))

	if result.Success {
		fmt.Println(color.GreenString("✓ Command executed successfully"))
	} else {
		fmt.Println(color.RedString("✗ Command failed"))
		if result.Error != "" {
			fmt.Printf("Error: %s\n", result.Error)
		}
	}

	fmt.Printf("Exit Code: %d\n", result.ExitCode)
	fmt.Printf("Duration: %v\n", result.Duration)

	if result.Output != "" {
		fmt.Println("\nOutput:")
		fmt.Println(result.Output)
	}

	fmt.Println(strings.Repeat("-", 50))
}

func runDocs(cmd *cobra.Command, args []string, rootCmd *cobra.Command, version string) error {
	// Create registry builder
	builder := agent.NewRegistryBuilder()

	// Build registry from root command
	registry := builder.BuildFromCobraCommand(rootCmd)

	// Add metadata
	registry.Metadata["binary"] = "iaac"
	registry.Metadata["version"] = version

	switch docsOutputFormat {
	case "json":
		// Save as JSON
		if err := builder.SaveToFile(docsOutputFile); err != nil {
			return fmt.Errorf("failed to save registry: %w", err)
		}
		fmt.Printf("Command registry saved to: %s\n", docsOutputFile)

	case "markdown":
		// Generate markdown
		markdown := builder.GenerateMarkdownDoc()

		// Change extension to .md
		if strings.HasSuffix(docsOutputFile, ".json") {
			docsOutputFile = docsOutputFile[:len(docsOutputFile)-5] + ".md"
		}

		if err := os.WriteFile(docsOutputFile, []byte(markdown), 0644); err != nil {
			return fmt.Errorf("failed to write markdown: %w", err)
		}
		fmt.Printf("Command documentation saved to: %s\n", docsOutputFile)

	default:
		return fmt.Errorf("unsupported format: %s", docsOutputFormat)
	}

	// Print summary
	fmt.Printf("\nDocumentation Summary:\n")
	fmt.Printf("- Total commands: %d\n", countCommands(registry.Commands))
	fmt.Printf("- Categories: %d\n", countCategories(registry.Commands))
	fmt.Printf("- Dangerous commands: %d\n", countDangerousCommands(registry.Commands))

	return nil
}

func countCommands(commands []agent.Command) int {
	count := len(commands)
	for _, cmd := range commands {
		count += countCommands(cmd.Subcommands)
	}
	return count
}

func countCategories(commands []agent.Command) int {
	categories := make(map[string]bool)
	var collectCategories func([]agent.Command)
	collectCategories = func(cmds []agent.Command) {
		for _, cmd := range cmds {
			if cmd.Category != "" {
				categories[cmd.Category] = true
			}
			collectCategories(cmd.Subcommands)
		}
	}
	collectCategories(commands)
	return len(categories)
}

func countDangerousCommands(commands []agent.Command) int {
	count := 0
	for _, cmd := range commands {
		if cmd.Dangerous {
			count++
		}
		count += countDangerousCommands(cmd.Subcommands)
	}
	return count
}
