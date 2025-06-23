// Package taskexec provides Task integration for executing Taskfile tasks
package taskexec

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
)

// Executor implements TaskExecutor interface using Task binary
type Executor struct {
	logger       interfaces.Logger
	taskfile     *Taskfile
	dir          string
	taskBinary   string
	taskfilePath string
}

// Taskfile represents a parsed Taskfile structure
type Taskfile struct {
	Version string                 `yaml:"version"`
	Tasks   map[string]Task        `yaml:"tasks"`
	Vars    map[string]interface{} `yaml:"vars"`
}

// Task represents a task in the Taskfile
type Task struct {
	Desc    string                 `yaml:"desc"`
	Summary string                 `yaml:"summary"`
	Cmds    []interface{}          `yaml:"cmds"`
	Deps    []interface{}          `yaml:"deps"`
	Vars    map[string]interface{} `yaml:"vars"`
	Silent  bool                   `yaml:"silent"`
}

// NewExecutor creates a new task executor
func NewExecutor(logger interfaces.Logger) *Executor {
	executor := &Executor{
		logger: logger,
	}

	// Find task binary
	if binary := findTaskBinary(); binary != "" {
		executor.taskBinary = binary
		logger.Debug("Found task binary: %s", binary)
	} else {
		logger.Warning("Task binary not found in PATH")
	}

	return executor
}

// findTaskBinary finds the task binary in the system
func findTaskBinary() string {
	// Check common locations for task binary
	possibleNames := []string{"task"}
	if runtime.GOOS == "windows" {
		possibleNames = append(possibleNames, "task.exe")
	}

	for _, name := range possibleNames {
		if path, err := exec.LookPath(name); err == nil {
			return path
		}
	}

	return ""
}

// LoadTaskfile loads and parses a Taskfile
func (e *Executor) LoadTaskfile(taskfilePath string) error {
	absPath, err := filepath.Abs(taskfilePath)
	if err != nil {
		return fmt.Errorf("failed to get absolute path: %w", err)
	}

	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		return fmt.Errorf("taskfile not found: %s", absPath)
	}

	// Read and parse the YAML file
	data, err := os.ReadFile(absPath)
	if err != nil {
		return fmt.Errorf("failed to read taskfile: %w", err)
	}

	var taskfile Taskfile
	if err := yaml.Unmarshal(data, &taskfile); err != nil {
		return fmt.Errorf("failed to parse taskfile: %w", err)
	}

	e.taskfile = &taskfile
	e.taskfilePath = absPath

	// Set working directory to project root, not taskfile directory
	e.dir = e.findProjectRoot(absPath)

	e.logger.Debug("Loaded taskfile: %s (working dir: %s)", absPath, e.dir)
	return nil
}

// ValidateDirectory validates that the directory contains required structure
func (e *Executor) ValidateDirectory(dir string) error {
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return fmt.Errorf("failed to get absolute path: %w", err)
	}

	// Check if directory exists
	if _, err := os.Stat(absDir); os.IsNotExist(err) {
		return fmt.Errorf("directory does not exist: %s", absDir)
	}

	// Try to find tasks directory in multiple locations
	var tasksDir string

	// Option 1: Check for tasks directory directly (when running from devops dir)
	directTasksDir := filepath.Join(absDir, "tasks")
	if _, err := os.Stat(directTasksDir); err == nil {
		tasksDir = directTasksDir
		e.logger.Debug("Found tasks directory at: %s", tasksDir)
	} else {
		// Option 2: Check for devops/tasks directory (when running from project root)
		devopsTasksDir := filepath.Join(absDir, "devops", "tasks")
		if _, err := os.Stat(devopsTasksDir); err == nil {
			tasksDir = devopsTasksDir
			e.logger.Debug("Found devops/tasks directory at: %s", tasksDir)
		} else {
			return fmt.Errorf("tasks directory not found. Looked for:\n- %s\n- %s", directTasksDir, devopsTasksDir)
		}
	}

	// Check for basic task files
	requiredDirs := []string{"build", "deploy"}
	for _, reqDir := range requiredDirs {
		dirPath := filepath.Join(tasksDir, reqDir)
		if _, err := os.Stat(dirPath); os.IsNotExist(err) {
			e.logger.Warning("Optional directory not found: %s", dirPath)
		}
	}

	e.logger.Info("Directory validation passed: %s (tasks dir: %s)", absDir, tasksDir)
	return nil
}

// ExecuteTask executes a specific task with given variables
func (e *Executor) ExecuteTask(ctx context.Context, taskName string, vars map[string]string) error {
	if e.taskfile == nil {
		return fmt.Errorf("no taskfile loaded")
	}

	// Find the task
	task, exists := e.taskfile.Tasks[taskName]
	if !exists {
		// Provide helpful suggestions for similar task names
		suggestions := e.findSimilarTasks(taskName)
		if len(suggestions) > 0 {
			return fmt.Errorf("task '%s' not found. Did you mean one of these?\n%s", taskName, strings.Join(suggestions, "\n"))
		}

		// List all available tasks if no similar ones found
		availableTasks := make([]string, 0, len(e.taskfile.Tasks))
		for name := range e.taskfile.Tasks {
			availableTasks = append(availableTasks, name)
		}
		return fmt.Errorf("task '%s' not found. Available tasks: %s", taskName, strings.Join(availableTasks, ", "))
	}

	e.logger.Info("Executing task: %s", taskName)

	// Use task binary if available, otherwise fall back to direct execution
	if e.taskBinary != "" {
		return e.executeWithTaskBinary(ctx, taskName, vars)
	}

	return e.executeDirectly(ctx, taskName, vars, task)
}

// executeWithTaskBinary executes task using the task binary
func (e *Executor) executeWithTaskBinary(ctx context.Context, taskName string, vars map[string]string) error {
	args := []string{"--taskfile", e.taskfilePath, "--dir", e.dir}

	// Add task name
	args = append(args, taskName)

	// Execute command
	cmd := exec.CommandContext(ctx, e.taskBinary, args...)
	cmd.Dir = e.dir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Set variables as environment variables (Task v3 way)
	if cmd.Env == nil {
		cmd.Env = os.Environ()
	}

	// Filter out problematic environment variables that can interfere with Go builds
	var cleanEnv []string
	for _, env := range cmd.Env {
		// Skip LDFLAGS that might contain linker-specific flags not compatible with Go
		if strings.HasPrefix(env, "LDFLAGS=") {
			e.logger.Debug("Filtering out potentially problematic LDFLAGS: %s", env)
			continue
		}
		// Also filter out other potentially problematic linker variables
		if strings.HasPrefix(env, "LD_") || strings.HasPrefix(env, "DYLD_") {
			e.logger.Debug("Filtering out potentially problematic env var: %s", strings.Split(env, "=")[0])
			continue
		}
		cleanEnv = append(cleanEnv, env)
	}
	cmd.Env = cleanEnv

	// Explicitly set LDFLAGS to empty to override any system defaults
	cmd.Env = append(cmd.Env, "LDFLAGS=")

	// Add user-provided variables
	for key, value := range vars {
		cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", key, value))
	}

	e.logger.Debug("Executing task binary from directory: %s", e.dir)

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to execute task '%s': %w", taskName, err)
	}

	e.logger.Success("Task completed: %s", taskName)
	return nil
}

// executeDirectly executes task commands directly (fallback)
func (e *Executor) executeDirectly(ctx context.Context, taskName string, vars map[string]string, task Task) error {
	e.logger.Warning("Executing task commands directly (task binary not available)")

	// Execute each command in the task
	for _, cmdInterface := range task.Cmds {
		// Convert command to string
		var cmdStr string
		if str, ok := cmdInterface.(string); ok {
			cmdStr = str
		} else {
			cmdStr = fmt.Sprintf("%v", cmdInterface)
		}

		// Simple variable substitution
		for key, value := range vars {
			cmdStr = strings.ReplaceAll(cmdStr, "{{."+key+"}}", value)
		}

		e.logger.Info("Running: %s", cmdStr)

		// Split command for execution
		parts := strings.Fields(cmdStr)
		if len(parts) == 0 {
			continue
		}

		cmd := exec.CommandContext(ctx, parts[0], parts[1:]...)
		cmd.Dir = e.dir
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr

		if err := cmd.Run(); err != nil {
			return fmt.Errorf("command failed: %s: %w", cmdStr, err)
		}
	}

	e.logger.Success("Task completed: %s", taskName)
	return nil
}

// ListTasks returns information about all available tasks
func (e *Executor) ListTasks() ([]interfaces.TaskInfo, error) {
	if e.taskfile == nil {
		return nil, fmt.Errorf("no taskfile loaded")
	}

	var tasks []interfaces.TaskInfo
	for name, task := range e.taskfile.Tasks {
		info := interfaces.TaskInfo{
			Name:        name,
			Description: task.Desc,
			Summary:     task.Summary,
			Vars:        convertInterfaceMapToStringMap(task.Vars),
			Deps:        convertInterfaceSliceToStringSlice(task.Deps),
		}

		tasks = append(tasks, info)
	}

	return tasks, nil
}

// convertInterfaceMapToStringMap converts map[string]interface{} to map[string]string
func convertInterfaceMapToStringMap(input map[string]interface{}) map[string]string {
	result := make(map[string]string)
	for key, value := range input {
		if str, ok := value.(string); ok {
			result[key] = str
		} else {
			result[key] = fmt.Sprintf("%v", value)
		}
	}
	return result
}

// convertInterfaceSliceToStringSlice converts []interface{} to []string
func convertInterfaceSliceToStringSlice(input []interface{}) []string {
	var result []string
	for _, item := range input {
		if str, ok := item.(string); ok {
			result = append(result, str)
		} else {
			result = append(result, fmt.Sprintf("%v", item))
		}
	}
	return result
}

// findSimilarTasks finds task names similar to the given input using basic string similarity
func (e *Executor) findSimilarTasks(input string) []string {
	var suggestions []string
	input = strings.ToLower(input)

	for taskName := range e.taskfile.Tasks {
		taskLower := strings.ToLower(taskName)

		// Exact substring match
		if strings.Contains(taskLower, input) || strings.Contains(input, taskLower) {
			suggestions = append(suggestions, fmt.Sprintf("  • %s", taskName))
			continue
		}

		// Check for common typos and similar patterns
		if e.isTypo(input, taskLower) {
			suggestions = append(suggestions, fmt.Sprintf("  • %s", taskName))
		}
	}

	return suggestions
}

// isTypo checks if two strings are likely typos of each other
func (e *Executor) isTypo(a, b string) bool {
	// Calculate simple edit distance (Levenshtein distance with limit of 2)
	return e.editDistance(a, b) <= 2
}

// editDistance calculates the edit distance between two strings (limited to 3 for performance)
func (e *Executor) editDistance(a, b string) int {
	if len(a) > len(b) {
		a, b = b, a
	}

	if len(b)-len(a) > 3 {
		return 4 // Return value > threshold to indicate no match
	}

	// Create matrix
	matrix := make([][]int, len(a)+1)
	for i := range matrix {
		matrix[i] = make([]int, len(b)+1)
	}

	// Initialize first row and column
	for i := 0; i <= len(a); i++ {
		matrix[i][0] = i
	}
	for j := 0; j <= len(b); j++ {
		matrix[0][j] = j
	}

	// Fill the matrix
	for i := 1; i <= len(a); i++ {
		for j := 1; j <= len(b); j++ {
			cost := 0
			if a[i-1] != b[j-1] {
				cost = 1
			}

			matrix[i][j] = min(
				matrix[i-1][j]+1,      // deletion
				matrix[i][j-1]+1,      // insertion
				matrix[i-1][j-1]+cost, // substitution
			)
		}
	}

	return matrix[len(a)][len(b)]
}

// min returns the minimum of three integers
func min(a, b, c int) int {
	if a < b {
		if a < c {
			return a
		}
		return c
	}
	if b < c {
		return b
	}
	return c
}

// abs returns the absolute value of an integer
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// findProjectRoot finds the correct project root directory for task execution
func (e *Executor) findProjectRoot(taskfilePath string) string {
	// Get the directory containing the taskfile
	taskfileDir := filepath.Dir(taskfilePath)

	// If taskfile is in devops/tasks/*, the project root should be the devops directory
	// (not the parent project) since we want to build/work within the devops tool
	if strings.Contains(taskfilePath, "devops/tasks/") {
		// Find the devops directory
		parts := strings.Split(taskfileDir, string(filepath.Separator))
		for i, part := range parts {
			if part == "devops" && i < len(parts)-1 && parts[i+1] == "tasks" {
				// Reconstruct path up to and including devops
				devopsRoot := filepath.Join(parts[:i+1]...)
				if !filepath.IsAbs(devopsRoot) && devopsRoot != "" {
					devopsRoot = "/" + devopsRoot
				}
				if devopsRoot == "" {
					devopsRoot = "/"
				}
				e.logger.Debug("Found project root (devops/tasks): %s", devopsRoot)
				return devopsRoot
			}
		}
	}

	// If taskfile is in tasks/* (direct tasks), the project root is the parent directory
	if strings.Contains(taskfilePath, "/tasks/") {
		parts := strings.Split(taskfileDir, string(filepath.Separator))
		for i, part := range parts {
			if part == "tasks" && i > 0 {
				// Reconstruct path up to tasks parent directory
				parentRoot := filepath.Join(parts[:i]...)
				if !filepath.IsAbs(parentRoot) && parentRoot != "" {
					parentRoot = "/" + parentRoot
				}
				if parentRoot == "" {
					parentRoot = "/"
				}
				e.logger.Debug("Found project root (tasks): %s", parentRoot)
				return parentRoot
			}
		}
	}

	// Fallback: use taskfile directory
	e.logger.Debug("Using taskfile directory as project root: %s", taskfileDir)
	return taskfileDir
}
