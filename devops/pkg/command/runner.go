// Package command provides command execution functionality
package command

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
)

// Runner implements the interfaces.CommandRunner interface
type Runner struct {
	logger interfaces.Logger
}

// NewRunner creates a new command runner
func NewRunner(logger interfaces.Logger) interfaces.CommandRunner {
	return &Runner{
		logger: logger,
	}
}

// Run executes a command
func (r *Runner) Run(ctx context.Context, name string, args ...string) error {
	cmd := exec.CommandContext(ctx, name, args...)

	r.logger.Debug("Running command: %s %s", name, strings.Join(args, " "))

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("command failed: %w\nOutput: %s", err, string(output))
	}

	return nil
}

// RunWithOutput executes a command and returns its output
func (r *Runner) RunWithOutput(ctx context.Context, name string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, name, args...)

	r.logger.Debug("Running command: %s %s", name, strings.Join(args, " "))

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		return "", fmt.Errorf("command failed: %w\nStderr: %s", err, stderr.String())
	}

	return strings.TrimSpace(stdout.String()), nil
}

// RunWithEnv executes a command with custom environment variables
func (r *Runner) RunWithEnv(ctx context.Context, env []string, name string, args ...string) error {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Env = append(cmd.Environ(), env...)

	r.logger.Debug("Running command with env: %s %s", name, strings.Join(args, " "))

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("command failed: %w\nOutput: %s", err, string(output))
	}

	return nil
}
