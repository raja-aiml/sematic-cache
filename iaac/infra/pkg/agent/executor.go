package agent

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
)

// InternalCommandExecutor executes commands directly using internal APIs
type InternalCommandExecutor struct {
	config      *Config
	auditLogger *AuditLogger
}

// NewInternalCommandExecutor creates a new internal command executor
func NewInternalCommandExecutor(config *Config) (*InternalCommandExecutor, error) {
	auditLogger, err := NewAuditLogger(config.AuditLogPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create audit logger: %w", err)
	}

	return &InternalCommandExecutor{
		config:      config,
		auditLogger: auditLogger,
	}, nil
}

// Execute runs a command internally
func (e *InternalCommandExecutor) Execute(ctx context.Context, interpreted *InterpretedCommand) (*ExecutionResult, error) {
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

	// Execute command
	startTime := time.Now()
	output, err := e.executeInternal(ctx, interpreted)
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
		ExitCode:   0,
		AuditTrail: auditEntry,
	}

	if err != nil {
		result.Error = err.Error()
		result.ExitCode = 1
	}

	return result, nil
}

// Validate checks if a command is safe to execute
func (e *InternalCommandExecutor) Validate(cmd *InterpretedCommand) error {
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

// executeInternal executes command using internal APIs
func (e *InternalCommandExecutor) executeInternal(ctx context.Context, interpreted *InterpretedCommand) (string, error) {
	// Get command factory
	factory, exists := e.config.CommandFactories[interpreted.Command]
	if !exists {
		return "", fmt.Errorf("unknown command: %s", interpreted.Command)
	}

	// Build cobra command
	cobraCmd := factory()

	// Build args
	args := []string{}
	if interpreted.Subcommand != "" {
		args = append(args, interpreted.Subcommand)
	}
	args = append(args, interpreted.Args...)

	// Set flags
	for key, value := range interpreted.Options {
		if err := cobraCmd.Flags().Set(key, value); err != nil {
			// Try with subcommand
			if interpreted.Subcommand != "" {
				if subCmd, _, err := cobraCmd.Find([]string{interpreted.Subcommand}); err == nil {
					if err := subCmd.Flags().Set(key, value); err != nil {
						return "", fmt.Errorf("failed to set flag --%s: %w", key, err)
					}
				}
			} else {
				return "", fmt.Errorf("failed to set flag --%s: %w", key, err)
			}
		}
	}

	// Capture output
	oldStdout := os.Stdout
	oldStderr := os.Stderr
	r, w, _ := os.Pipe()
	os.Stdout = w
	os.Stderr = w

	// Execute command with timeout
	done := make(chan error, 1)
	var output bytes.Buffer

	go func() {
		// Execute the command
		cobraCmd.SetArgs(args)
		err := cobraCmd.Execute()

		w.Close()
		done <- err
	}()

	// Read output
	go func() {
		buf := make([]byte, 1024)
		for {
			n, err := r.Read(buf)
			if n > 0 {
				output.Write(buf[:n])
			}
			if err != nil {
				break
			}
		}
	}()

	// Wait for completion or timeout
	timeout := e.config.CommandTimeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	select {
	case err := <-done:
		os.Stdout = oldStdout
		os.Stderr = oldStderr
		return output.String(), err
	case <-time.After(timeout):
		os.Stdout = oldStdout
		os.Stderr = oldStderr
		return output.String(), fmt.Errorf("command timed out after %v", timeout)
	case <-ctx.Done():
		os.Stdout = oldStdout
		os.Stderr = oldStderr
		return output.String(), ctx.Err()
	}
}

// confirmExecution prompts user for confirmation
func (e *InternalCommandExecutor) confirmExecution(cmd *InterpretedCommand) bool {
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
func (e *InternalCommandExecutor) formatCommand(cmd *InterpretedCommand) string {
	parts := []string{"iaac", cmd.Command}

	if cmd.Subcommand != "" {
		parts = append(parts, cmd.Subcommand)
	}

	for _, arg := range cmd.Args {
		parts = append(parts, arg)
	}

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
func (e *InternalCommandExecutor) matchesPattern(command, pattern string) bool {
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
func (e *InternalCommandExecutor) getCurrentUser() string {
	if user := os.Getenv("USER"); user != "" {
		return user
	}
	if user := os.Getenv("USERNAME"); user != "" {
		return user
	}
	return "unknown"
}
