package agent

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/fatih/color"
)

// CLIAgent implements the NLP-powered CLI agent
type CLIAgent struct {
	config   *Config
	nlp      NLPEngine
	executor CommandExecutor
	registry *CommandRegistry
}

// NewCLIAgent creates a new CLI agent
func NewCLIAgent(config *Config) (*CLIAgent, error) {
	// Load or create command registry
	registry, err := loadOrCreateRegistry(config)
	if err != nil {
		return nil, fmt.Errorf("failed to load command registry: %w", err)
	}

	// Create NLP engine
	nlp, err := NewOpenAINLPEngine(config.OpenAIKey, config.OpenAIModel, config.OpenAIMaxTokens)
	if err != nil {
		return nil, fmt.Errorf("failed to create NLP engine: %w", err)
	}

	// Create executor - use internal executor
	executor, err := NewInternalCommandExecutor(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create command executor: %w", err)
	}

	return &CLIAgent{
		config:   config,
		nlp:      nlp,
		executor: executor,
		registry: registry,
	}, nil
}

// ProcessQuery takes a natural language query and returns an executable command
func (a *CLIAgent) ProcessQuery(ctx context.Context, query string) (*CommandResult, error) {
	// Clean up the query
	query = strings.TrimSpace(query)
	if query == "" {
		return &CommandResult{
			Success: false,
			Error:   "Empty query provided",
		}, nil
	}

	// Use NLP to interpret the query
	interpreted, err := a.nlp.Interpret(ctx, query, a.registry)
	if err != nil {
		// Try to provide helpful suggestions
		suggestions := a.getSuggestions(query)
		return &CommandResult{
			Success:     false,
			Error:       fmt.Sprintf("Failed to interpret query: %v", err),
			Suggestions: suggestions,
		}, nil
	}

	// Validate the interpreted command
	if err := a.executor.Validate(interpreted); err != nil {
		return &CommandResult{
			Success: false,
			Command: interpreted,
			Error:   fmt.Sprintf("Command validation failed: %v", err),
		}, nil
	}

	return &CommandResult{
		Success: true,
		Command: interpreted,
	}, nil
}

// ExecuteQuery processes and executes a natural language query
func (a *CLIAgent) ExecuteQuery(ctx context.Context, query string) (*ExecutionResult, error) {
	// First process the query
	result, err := a.ProcessQuery(ctx, query)
	if err != nil {
		return nil, err
	}

	if !result.Success {
		return &ExecutionResult{
			Success: false,
			Error:   result.Error,
		}, nil
	}

	// Execute the command
	return a.executor.Execute(ctx, result.Command)
}

// GetCommandRegistry returns the available commands documentation
func (a *CLIAgent) GetCommandRegistry() *CommandRegistry {
	return a.registry
}

// StartInteractive starts an interactive session
func (a *CLIAgent) StartInteractive(ctx context.Context) error {
	// Print welcome message
	a.printWelcome()

	// Load history if configured
	history := a.loadHistory()

	reader := bufio.NewReader(os.Stdin)
	prompt := a.config.InteractivePrompt
	if prompt == "" {
		prompt = "iaac> "
	}

	for {
		// Print prompt
		fmt.Print(color.CyanString(prompt))

		// Read input
		input, err := reader.ReadString('\n')
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return fmt.Errorf("failed to read input: %w", err)
		}

		input = strings.TrimSpace(input)

		// Handle special commands
		if a.handleSpecialCommand(input) {
			continue
		}

		// Skip empty input
		if input == "" {
			continue
		}

		// Add to history
		history = append(history, input)

		// Process the query
		a.processInteractiveQuery(ctx, input)
	}

	// Save history
	a.saveHistory(history)

	fmt.Println("\nGoodbye!")
	return nil
}

// printWelcome prints the welcome message
func (a *CLIAgent) printWelcome() {
	fmt.Println(color.GreenString(`
╔═══════════════════════════════════════════════╗
║         Infrastructure Agent (iaac)           ║
║                                               ║
║  Natural Language Infrastructure Management   ║
╚═══════════════════════════════════════════════╝
`))
	fmt.Println("Type 'help' for assistance or 'exit' to quit.")
	fmt.Println()
}

// handleSpecialCommand handles special interactive commands
func (a *CLIAgent) handleSpecialCommand(input string) bool {
	switch strings.ToLower(input) {
	case "help", "?":
		a.printHelp()
		return true
	case "exit", "quit", "bye":
		return false
	case "commands":
		a.printCommands()
		return true
	case "examples":
		a.printExamples()
		return true
	case "clear":
		fmt.Print("\033[H\033[2J")
		return true
	}

	return false
}

// processInteractiveQuery processes a query in interactive mode
func (a *CLIAgent) processInteractiveQuery(ctx context.Context, query string) {
	// Show processing indicator
	fmt.Print(color.YellowString("Processing... "))

	// Process the query
	result, err := a.ProcessQuery(ctx, query)
	if err != nil {
		fmt.Println(color.RedString("✗"))
		fmt.Printf("Error: %v\n", err)
		return
	}

	if !result.Success {
		fmt.Println(color.RedString("✗"))
		fmt.Printf("Error: %s\n", result.Error)
		if len(result.Suggestions) > 0 {
			fmt.Println("\nSuggestions:")
			for _, suggestion := range result.Suggestions {
				fmt.Printf("  • %s\n", suggestion)
			}
		}
		return
	}

	fmt.Println(color.GreenString("✓"))

	// Display the interpreted command
	cmd := result.Command
	fmt.Println("\nInterpreted Command:")
	fmt.Printf("  %s\n", color.CyanString(a.formatCommand(cmd)))
	fmt.Printf("  Confidence: %.0f%%\n", cmd.Confidence*100)
	if cmd.Explanation != "" {
		fmt.Printf("  Explanation: %s\n", cmd.Explanation)
	}

	// Check if it's dangerous
	if cmd.Dangerous {
		fmt.Println(color.YellowString("\n⚠️  This is a dangerous command!"))
	}

	// Ask for confirmation
	if !a.confirmInteractiveExecution() {
		fmt.Println("Command cancelled.")
		return
	}

	// Execute the command
	fmt.Println("\nExecuting...")
	execResult, err := a.executor.Execute(ctx, cmd)
	if err != nil {
		fmt.Printf("Execution error: %v\n", err)
		return
	}

	// Display results
	a.displayExecutionResult(execResult)
}

// confirmInteractiveExecution asks for confirmation in interactive mode
func (a *CLIAgent) confirmInteractiveExecution() bool {
	fmt.Print("\nExecute this command? (y/n): ")

	reader := bufio.NewReader(os.Stdin)
	response, err := reader.ReadString('\n')
	if err != nil {
		return false
	}

	response = strings.TrimSpace(strings.ToLower(response))
	return response == "y" || response == "yes"
}

// displayExecutionResult displays the execution result
func (a *CLIAgent) displayExecutionResult(result *ExecutionResult) {
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

// formatCommand formats a command for display
func (a *CLIAgent) formatCommand(cmd *InterpretedCommand) string {
	parts := []string{cmd.Command}

	if cmd.Subcommand != "" {
		parts = append(parts, cmd.Subcommand)
	}

	parts = append(parts, cmd.Args...)

	for key, value := range cmd.Options {
		if len(key) == 1 {
			parts = append(parts, fmt.Sprintf("-%s", key))
		} else {
			parts = append(parts, fmt.Sprintf("--%s", key))
		}
		if value != "true" && value != "" {
			parts = append(parts, value)
		}
	}

	return strings.Join(parts, " ")
}

// getSuggestions provides command suggestions based on query
func (a *CLIAgent) getSuggestions(query string) []string {
	suggestions := []string{}
	queryLower := strings.ToLower(query)

	// Check for common patterns
	patterns := map[string][]string{
		"list":   {"Try: 'show all clusters'", "Try: 'list blueprints'"},
		"create": {"Try: 'create a new cluster named dev'", "Try: 'create cluster with 3 nodes'"},
		"delete": {"Try: 'delete cluster dev'", "Try: 'remove cluster named test'"},
		"deploy": {"Try: 'deploy nginx to cluster'", "Try: 'apply manifest from file.yaml'"},
	}

	for pattern, suggs := range patterns {
		if strings.Contains(queryLower, pattern) {
			suggestions = append(suggestions, suggs...)
		}
	}

	if len(suggestions) == 0 {
		suggestions = append(suggestions, "Try: 'help' to see available commands")
		suggestions = append(suggestions, "Try: 'examples' to see example queries")
	}

	return suggestions
}

// printHelp prints help information
func (a *CLIAgent) printHelp() {
	fmt.Print(`
Available Commands:
  help, ?     - Show this help message
  commands    - List all available commands
  examples    - Show example queries
  clear       - Clear the screen
  exit, quit  - Exit the interactive session

Natural Language Queries:
  You can use natural language to describe what you want to do.
  
  Examples:
    • "Create a new cluster with 3 nodes"
    • "Show me all running clusters"
    • "Delete the test cluster"
    • "Deploy nginx to the production cluster"
    
Tips:
  - Be specific about what you want to do
  - Include names, numbers, and options in your query
  - The agent will ask for confirmation before executing commands
`)
}

// printCommands prints available commands
func (a *CLIAgent) printCommands() {
	fmt.Println("\nAvailable Commands:")
	for _, cmd := range a.registry.Commands {
		a.printCommand(cmd, "  ")
	}
}

// printCommand recursively prints command information
func (a *CLIAgent) printCommand(cmd Command, indent string) {
	fmt.Printf("%s%s - %s\n", indent, color.CyanString(cmd.Name), cmd.Description)

	if cmd.Dangerous {
		fmt.Printf("%s  %s\n", indent, color.YellowString("[DANGEROUS]"))
	}

	for _, subcmd := range cmd.Subcommands {
		a.printCommand(subcmd, indent+"  ")
	}
}

// printExamples prints example queries
func (a *CLIAgent) printExamples() {
	examples := []struct {
		query       string
		description string
	}{
		{"create a new k3d cluster called development", "Creates a development cluster"},
		{"show me all clusters", "Lists all k3d clusters"},
		{"delete the test cluster", "Removes the test cluster"},
		{"create a 3 node cluster with k3s version 1.28", "Creates a multi-node cluster"},
		{"deploy nginx to the dev cluster", "Deploys nginx application"},
		{"show cluster details for production", "Gets information about a specific cluster"},
		{"validate the blueprint file config.yaml", "Validates a blueprint configuration"},
		{"apply kubernetes manifest from app.yaml", "Applies a Kubernetes manifest"},
	}

	fmt.Println("\nExample Natural Language Queries:")
	for _, ex := range examples {
		fmt.Printf("\n  Query: %s\n", color.GreenString(ex.query))
		fmt.Printf("  Result: %s\n", ex.description)
	}
}

// loadHistory loads command history
func (a *CLIAgent) loadHistory() []string {
	if a.config.HistoryFile == "" {
		return []string{}
	}

	data, err := os.ReadFile(a.config.HistoryFile)
	if err != nil {
		return []string{}
	}

	return strings.Split(string(data), "\n")
}

// saveHistory saves command history
func (a *CLIAgent) saveHistory(history []string) {
	if a.config.HistoryFile == "" {
		return
	}

	// Keep last 1000 commands
	if len(history) > 1000 {
		history = history[len(history)-1000:]
	}

	data := strings.Join(history, "\n")
	if err := os.WriteFile(a.config.HistoryFile, []byte(data), 0644); err != nil {
		// Log error but don't fail - history is not critical
		fmt.Fprintf(os.Stderr, "Warning: Failed to save history: %v\n", err)
	}
}

// loadOrCreateRegistry loads or creates a command registry
func loadOrCreateRegistry(config *Config) (*CommandRegistry, error) {
	// Try to load from file first
	registryPath := filepath.Join(filepath.Dir(config.AuditLogPath), "commands.json")
	if registry, err := LoadRegistryFromFile(registryPath); err == nil {
		return registry, nil
	}

	// Create default registry
	return CreateDefaultRegistry(), nil
}
