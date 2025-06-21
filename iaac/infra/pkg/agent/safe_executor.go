package agent

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/google/uuid"
)

// CommandBuilder builds command arguments from interpreted commands
type CommandBuilder interface {
	Build(cmd *InterpretedCommand) ([]string, error)
}

// DefaultCommandBuilder builds commands for execution
type DefaultCommandBuilder struct {
	binaryPath string
}

// NewDefaultCommandBuilder creates a new default command builder
func NewDefaultCommandBuilder(binaryPath string) *DefaultCommandBuilder {
	return &DefaultCommandBuilder{
		binaryPath: binaryPath,
	}
}

// Build constructs command arguments from an interpreted command
func (b *DefaultCommandBuilder) Build(cmd *InterpretedCommand) ([]string, error) {
	args := []string{b.binaryPath, cmd.Command}

	if cmd.Subcommand != "" {
		args = append(args, cmd.Subcommand)
	}

	args = append(args, cmd.Args...)

	for key, value := range cmd.Options {
		if len(key) == 1 {
			args = append(args, fmt.Sprintf("-%s", key))
		} else {
			args = append(args, fmt.Sprintf("--%s", key))
		}
		if value != "true" && value != "" {
			args = append(args, value)
		}
	}

	return args, nil
}

// SafeCommandExecutor executes commands safely with validation and auditing
type SafeCommandExecutor struct {
	config         *Config
	auditLogger    *AuditLogger
	commandBuilder CommandBuilder
}

// NewSafeCommandExecutor creates a new safe command executor
func NewSafeCommandExecutor(config *Config) (*SafeCommandExecutor, error) {
	auditLogger, err := NewAuditLogger(config.AuditLogPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create audit logger: %w", err)
	}

	return &SafeCommandExecutor{
		config:         config,
		auditLogger:    auditLogger,
		commandBuilder: NewDefaultCommandBuilder("iaac"),
	}, nil
}

// Execute runs a command safely with validation and auditing
func (e *SafeCommandExecutor) Execute(ctx context.Context, interpreted *InterpretedCommand) (*ExecutionResult, error) {
	// Validate command first
	if err := e.Validate(interpreted); err != nil {
		return nil, err
	}

	// Check if confirmation is required
	if e.config.RequireConfirmation && interpreted.Dangerous {
		if !e.confirmExecution(interpreted) {
			return &ExecutionResult{
				Success: false,
				Command: e.formatCommand(interpreted),
				Error:   "User cancelled execution",
			}, nil
		}
	}

	// Create audit entry
	auditEntry := &AuditEntry{
		ID:        uuid.New().String(),
		Timestamp: time.Now(),
		User:      e.getCurrentUser(),
		Query:     interpreted.Query,
		Command:   e.formatCommand(interpreted),
	}

	// Build command arguments
	cmdArgs, err := e.commandBuilder.Build(interpreted)
	if err != nil {
		return nil, fmt.Errorf("failed to build command: %w", err)
	}

	// Execute command
	startTime := time.Now()
	output, exitCode, err := e.executeWithTimeout(ctx, cmdArgs)
	duration := time.Since(startTime)

	// Update audit entry
	auditEntry.Duration = duration
	auditEntry.Success = err == nil
	if err != nil {
		auditEntry.Error = err.Error()
	}

	// Log audit entry
	if logErr := e.auditLogger.Log(auditEntry); logErr != nil {
		// Don't fail execution if audit logging fails
		fmt.Fprintf(os.Stderr, "Warning: Failed to log audit entry: %v\n", logErr)
	}

	result := &ExecutionResult{
		Success:    err == nil,
		Command:    e.formatCommand(interpreted),
		Output:     output,
		Duration:   duration,
		ExitCode:   exitCode,
		AuditTrail: auditEntry,
	}

	if err != nil {
		result.Error = err.Error()
	}

	return result, nil
}

// Validate checks if a command is safe to execute
func (e *SafeCommandExecutor) Validate(cmd *InterpretedCommand) error {
	// Check whitelist
	if len(e.config.CommandWhitelist) > 0 {
		allowed := false
		for _, pattern := range e.config.CommandWhitelist {
			if e.matchesPattern(cmd.Command, pattern) {
				allowed = true
				break
			}
		}
		if !allowed {
			return fmt.Errorf("command '%s' is not in whitelist", cmd.Command)
		}
	}

	// Check blacklist
	for _, pattern := range e.config.CommandBlacklist {
		if e.matchesPattern(cmd.Command, pattern) {
			return fmt.Errorf("command '%s' is blacklisted", cmd.Command)
		}
	}

	// Check dangerous commands
	if cmd.Dangerous && !e.config.EnableDangerousCommands {
		return fmt.Errorf("dangerous commands are disabled")
	}

	return nil
}

// executeWithTimeout executes a command with a timeout
func (e *SafeCommandExecutor) executeWithTimeout(ctx context.Context, cmdArgs []string) (string, int, error) {
	// Create timeout context if not already set
	timeout := e.config.CommandTimeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Create command with timeout context
	cmd := exec.CommandContext(ctx, cmdArgs[0], cmdArgs[1:]...)

	// Set up output capture
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// Execute command
	err := cmd.Run()

	// Get output
	output := stdout.String()
	if stderr.Len() > 0 {
		if output != "" {
			output += "\n"
		}
		output += stderr.String()
	}

	// Get exit code
	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else {
			exitCode = -1
		}

		// Check if it was a timeout
		if ctx.Err() == context.DeadlineExceeded {
			return output, exitCode, fmt.Errorf("command timed out after %v", timeout)
		}
	}

	return output, exitCode, err
}

// confirmExecution prompts user for confirmation
func (e *SafeCommandExecutor) confirmExecution(cmd *InterpretedCommand) bool {
	fmt.Printf("\n⚠️  Warning: You are about to execute a dangerous command:\n")
	fmt.Printf("Command: %s\n", e.formatCommand(cmd))
	fmt.Printf("Explanation: %s\n\n", cmd.Explanation)
	fmt.Print("Do you want to continue? (yes/no): ")

	var response string
	fmt.Scanln(&response)

	response = strings.TrimSpace(strings.ToLower(response))
	return response == "yes" || response == "y"
}

// formatCommand formats an interpreted command for display
func (e *SafeCommandExecutor) formatCommand(cmd *InterpretedCommand) string {
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

// matchesPattern checks if a command matches a pattern
func (e *SafeCommandExecutor) matchesPattern(command, pattern string) bool {
	// Simple glob matching
	if pattern == "*" {
		return true
	}

	if strings.HasSuffix(pattern, "*") {
		prefix := pattern[:len(pattern)-1]
		return strings.HasPrefix(command, prefix)
	}

	return command == pattern
}

// getCurrentUser returns the current system user
func (e *SafeCommandExecutor) getCurrentUser() string {
	if user := os.Getenv("USER"); user != "" {
		return user
	}
	if user := os.Getenv("USERNAME"); user != "" {
		return user
	}
	return "unknown"
}
