package taskcommand

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
	"github.com/raja-aiml/sematic-cache/devops/pkg/logger"
	"github.com/raja-aiml/sematic-cache/devops/pkg/taskexec"
)

// TaskCommand defines the interface for task-based commands
type TaskCommand interface {
	RunTask(taskfilePath string, args []string, vars map[string]string, workDir string) error
	ListTasks(executor interfaces.TaskExecutor, taskfilePath string) error
	GetSubCommand(taskfilePath string) string
}

// BaseTaskCommand provides common functionality for task-based commands
type BaseTaskCommand struct {
	logger interfaces.Logger
}

// NewBaseTaskCommand creates a new base task command
func NewBaseTaskCommand() *BaseTaskCommand {
	return &BaseTaskCommand{
		logger: logger.NewWithOptions(logger.InfoLevel, true),
	}
}

// RunTask executes a task from the specified taskfile
func (btc *BaseTaskCommand) RunTask(taskfilePath string, args []string, vars map[string]string, workDir string) error {
	// Determine working directory
	if workDir == "" {
		var err error
		workDir, err = os.Getwd()
		if err != nil {
			return fmt.Errorf("failed to get current directory: %w", err)
		}
	}

	// Create task executor
	executor := taskexec.NewExecutor(btc.logger)

	// Validate directory structure
	if err := executor.ValidateDirectory(workDir); err != nil {
		return fmt.Errorf("directory validation failed: %w", err)
	}

	// Find the tasks directory and load the specific taskfile
	fullTaskfilePath, err := btc.findTaskfilePath(workDir, taskfilePath)
	if err != nil {
		return fmt.Errorf("failed to find taskfile: %w", err)
	}

	if err := executor.LoadTaskfile(fullTaskfilePath); err != nil {
		return fmt.Errorf("failed to load taskfile: %w", err)
	}

	// If no task specified, list available tasks
	if len(args) == 0 {
		return btc.ListTasks(executor, taskfilePath)
	}

	// Execute the specified task
	taskName := args[0]
	ctx := context.Background()

	btc.logger.Info("Executing task '%s' from %s", taskName, taskfilePath)
	if err := executor.ExecuteTask(ctx, taskName, vars); err != nil {
		return fmt.Errorf("task execution failed: %w", err)
	}

	return nil
}

// ListTasks displays available tasks from the taskfile
func (btc *BaseTaskCommand) ListTasks(executor interfaces.TaskExecutor, taskfilePath string) error {
	tasks, err := executor.ListTasks()
	if err != nil {
		return fmt.Errorf("failed to list tasks: %w", err)
	}

	fmt.Printf("📋 Available tasks in %s:\n\n", taskfilePath)
	for _, task := range tasks {
		fmt.Printf("• %s", task.Name)
		if task.Description != "" {
			fmt.Printf(" - %s", task.Description)
		}
		fmt.Println()

		if task.Summary != "" {
			fmt.Printf("  %s\n", task.Summary)
		}

		if len(task.Deps) > 0 {
			fmt.Printf("  Dependencies: %v\n", task.Deps)
		}
		fmt.Println()
	}

	fmt.Println("Usage:")
	fmt.Printf("  devops %s %s <task>\n", btc.getCommandCategory(taskfilePath), btc.GetSubCommand(taskfilePath))
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Printf("  devops %s %s info\n", btc.getCommandCategory(taskfilePath), btc.GetSubCommand(taskfilePath))
	fmt.Printf("  devops %s %s --vars KEY=value info\n", btc.getCommandCategory(taskfilePath), btc.GetSubCommand(taskfilePath))

	return nil
}

// GetSubCommand returns the subcommand name based on taskfile path
func (btc *BaseTaskCommand) GetSubCommand(taskfilePath string) string {
	base := filepath.Base(taskfilePath)
	switch base {
	case "go.yaml":
		return "go"
	case "docker.yaml":
		return "docker"
	case "k8s.yaml":
		return "k8s"
	case "k3d.yaml":
		return "k3d"
	case "helm.yaml":
		return "helm"
	case "security.yaml":
		return "security"
	default:
		return "unknown"
	}
}

// getCommandCategory returns the main command category based on taskfile path
func (btc *BaseTaskCommand) getCommandCategory(taskfilePath string) string {
	dir := filepath.Dir(taskfilePath)
	return filepath.Base(dir)
}

// findTaskfilePath finds the correct path to a taskfile based on directory structure
func (btc *BaseTaskCommand) findTaskfilePath(workDir, taskfilePath string) (string, error) {
	// Try to find tasks directory in multiple locations

	// Option 1: Check for tasks directory directly (when running from devops dir)
	directTasksPath := filepath.Join(workDir, "tasks", taskfilePath)
	if _, err := os.Stat(directTasksPath); err == nil {
		btc.logger.Debug("Found taskfile at: %s", directTasksPath)
		return directTasksPath, nil
	}

	// Option 2: Check for devops/tasks directory (when running from project root)
	devopsTasksPath := filepath.Join(workDir, "devops", "tasks", taskfilePath)
	if _, err := os.Stat(devopsTasksPath); err == nil {
		btc.logger.Debug("Found taskfile at: %s", devopsTasksPath)
		return devopsTasksPath, nil
	}

	return "", fmt.Errorf("taskfile not found. Looked for:\n- %s\n- %s", directTasksPath, devopsTasksPath)
}
