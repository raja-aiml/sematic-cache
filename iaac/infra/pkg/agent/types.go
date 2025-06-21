package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"
)

// Agent represents the NLP-powered CLI agent
type Agent interface {
	// ProcessQuery takes a natural language query and returns an executable command
	ProcessQuery(ctx context.Context, query string) (*CommandResult, error)

	// ExecuteQuery processes and executes a natural language query
	ExecuteQuery(ctx context.Context, query string) (*ExecutionResult, error)

	// GetCommandRegistry returns the available commands documentation
	GetCommandRegistry() *CommandRegistry

	// StartInteractive starts an interactive session
	StartInteractive(ctx context.Context) error
}

// NLPEngine handles natural language processing
type NLPEngine interface {
	// Interpret converts natural language to structured command
	Interpret(ctx context.Context, query string, registry *CommandRegistry) (*InterpretedCommand, error)

	// GenerateDocumentation creates human-readable documentation
	GenerateDocumentation(commands []Command) (string, error)
}

// CommandExecutor handles safe command execution
type CommandExecutor interface {
	// Execute runs a command safely
	Execute(ctx context.Context, cmd *InterpretedCommand) (*ExecutionResult, error)

	// Validate checks if a command is safe to execute
	Validate(cmd *InterpretedCommand) error
}

// Command represents a CLI command that can be executed
type Command struct {
	Name                 string    `json:"name"`
	Description          string    `json:"description"`
	Category             string    `json:"category"`
	Subcommands          []Command `json:"subcommands,omitempty"`
	Options              []Option  `json:"options,omitempty"`
	Examples             []Example `json:"examples,omitempty"`
	Dangerous            bool      `json:"dangerous"`
	RequiresConfirmation bool      `json:"requires_confirmation"`
}

// Option represents a command-line option
type Option struct {
	Name        string   `json:"name"`
	Shorthand   string   `json:"shorthand,omitempty"`
	Type        string   `json:"type"`
	Description string   `json:"description"`
	Required    bool     `json:"required"`
	Default     string   `json:"default,omitempty"`
	Choices     []string `json:"choices,omitempty"`
}

// Example shows how to use a command
type Example struct {
	Command     string `json:"command"`
	Description string `json:"description"`
	Output      string `json:"output,omitempty"`
}

// CommandRegistry holds all available commands
type CommandRegistry struct {
	Commands    []Command         `json:"commands"`
	Version     string            `json:"version"`
	GeneratedAt time.Time         `json:"generated_at"`
	Metadata    map[string]string `json:"metadata"`
}

// InterpretedCommand represents a parsed natural language query
type InterpretedCommand struct {
	Query       string            `json:"query"`
	Command     string            `json:"command"`
	Subcommand  string            `json:"subcommand,omitempty"`
	Args        []string          `json:"args"`
	Options     map[string]string `json:"options"`
	Confidence  float64           `json:"confidence"`
	Explanation string            `json:"explanation"`
	Dangerous   bool              `json:"dangerous"`
}

// CommandResult represents the result of query processing
type CommandResult struct {
	Success     bool                `json:"success"`
	Command     *InterpretedCommand `json:"command,omitempty"`
	Error       string              `json:"error,omitempty"`
	Suggestions []string            `json:"suggestions,omitempty"`
}

// ExecutionResult represents the result of command execution
type ExecutionResult struct {
	Success    bool          `json:"success"`
	Command    string        `json:"command"`
	Output     string        `json:"output"`
	Error      string        `json:"error,omitempty"`
	Duration   time.Duration `json:"duration"`
	ExitCode   int           `json:"exit_code"`
	AuditTrail *AuditEntry   `json:"audit_trail"`
}

// AuditEntry records command execution for audit purposes
type AuditEntry struct {
	ID        string        `json:"id"`
	Timestamp time.Time     `json:"timestamp"`
	User      string        `json:"user"`
	Query     string        `json:"query"`
	Command   string        `json:"command"`
	Success   bool          `json:"success"`
	Error     string        `json:"error,omitempty"`
	Duration  time.Duration `json:"duration"`
}

// Config holds agent configuration
type Config struct {
	// OpenAI configuration
	OpenAIKey       string `json:"openai_key" env:"OPENAI_API_KEY"`
	OpenAIModel     string `json:"openai_model" env:"OPENAI_MODEL"`
	OpenAIMaxTokens int    `json:"openai_max_tokens" env:"OPENAI_MAX_TOKENS"`

	// Safety configuration
	EnableDangerousCommands bool     `json:"enable_dangerous_commands"`
	RequireConfirmation     bool     `json:"require_confirmation"`
	CommandWhitelist        []string `json:"command_whitelist"`
	CommandBlacklist        []string `json:"command_blacklist"`

	// Execution configuration
	CommandTimeout time.Duration `json:"command_timeout"`
	MaxRetries     int           `json:"max_retries"`
	AuditLogPath   string        `json:"audit_log_path"`

	// Interactive mode configuration
	InteractivePrompt  string `json:"interactive_prompt"`
	HistoryFile        string `json:"history_file"`
	EnableAutoComplete bool   `json:"enable_auto_complete"`

	// Command factories
	CommandFactories map[string]func() *cobra.Command `json:"-"`
}

// AuditLogger handles logging of command executions for audit purposes
type AuditLogger struct {
	filePath string
}

// NewAuditLogger creates a new audit logger
func NewAuditLogger(filePath string) (*AuditLogger, error) {
	// Ensure directory exists
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create audit log directory: %w", err)
	}

	return &AuditLogger{
		filePath: filePath,
	}, nil
}

// Log writes an audit entry to the log file
func (l *AuditLogger) Log(entry *AuditEntry) error {
	// Open file in append mode
	file, err := os.OpenFile(l.filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open audit log file: %w", err)
	}
	defer file.Close()

	// Encode entry as JSON
	data, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("failed to marshal audit entry: %w", err)
	}

	// Write to file with newline
	if _, err := file.Write(append(data, '\n')); err != nil {
		return fmt.Errorf("failed to write audit entry: %w", err)
	}

	return nil
}
