package cmd

import (
	"github.com/spf13/cobra"

	"github.com/raja-aiml/sematic-cache/devops/pkg/taskcommand"
)

var (
	buildDir     string
	buildVars    map[string]string
	buildTaskCmd taskcommand.TaskCommand
)

// buildCmd represents the build command
var buildCmd = &cobra.Command{
	Use:   "build",
	Short: "Build tasks for the project",
	Long: `Execute build-related tasks using the Task SDK.

Supports sub-commands:
- go: Go build tasks (build, test, lint, etc.)
- docker: Docker image tasks (build, push, scan, etc.)

Examples:
  devops build go --vars BINARY_NAME=myapp
  devops build docker --vars IMAGE_NAME=myimage
  devops build go --dir=/path/to/project`,
}

// buildGoCmd represents the build go command
var buildGoCmd = &cobra.Command{
	Use:   "go [task]",
	Short: "Execute Go build tasks",
	Long: `Execute Go build tasks from devops/tasks/build/go.yaml

Available tasks will be loaded from the Taskfile.

Examples:
  devops build go               # List available tasks
  devops build go build         # Execute go:build task
  devops build go test          # Execute go:test task
  devops build go lint          # Execute go:lint task`,
	RunE: runBuildGo,
}

// buildDockerCmd represents the build docker command
var buildDockerCmd = &cobra.Command{
	Use:   "docker [task]",
	Short: "Execute Docker build tasks",
	Long: `Execute Docker build tasks from devops/tasks/build/docker.yaml

Available tasks will be loaded from the Taskfile.

Examples:
  devops build docker           # List available tasks
  devops build docker build    # Execute docker:build task
  devops build docker push     # Execute docker:push task
  devops build docker scan     # Execute docker:scan task`,
	RunE: runBuildDocker,
}

func runBuildGo(cmd *cobra.Command, args []string) error {
	return buildTaskCmd.RunTask("build/go.yaml", args, buildVars, buildDir)
}

func runBuildDocker(cmd *cobra.Command, args []string) error {
	return buildTaskCmd.RunTask("build/docker.yaml", args, buildVars, buildDir)
}

func init() {
	// Initialize task command interface
	buildTaskCmd = taskcommand.NewBaseTaskCommand()

	// Add build command to root
	rootCmd.AddCommand(buildCmd)

	// Add subcommands to build
	buildCmd.AddCommand(buildGoCmd)
	buildCmd.AddCommand(buildDockerCmd)

	// Add flags
	buildCmd.PersistentFlags().StringVar(&buildDir, "dir", "", "Project directory (default: current directory)")
	buildCmd.PersistentFlags().StringToStringVar(&buildVars, "vars", map[string]string{}, "Task variables (key=value)")
}
