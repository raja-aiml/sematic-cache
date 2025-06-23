package cmd

import (
	"github.com/spf13/cobra"

	"github.com/raja-aiml/sematic-cache/devops/pkg/taskcommand"
)

var (
	qualityDir     string
	qualityVars    map[string]string
	qualityTaskCmd taskcommand.TaskCommand
)

// qualityCmd represents the quality command
var qualityCmd = &cobra.Command{
	Use:   "quality",
	Short: "Code quality and security tasks",
	Long: `Execute code quality and security-related tasks using the Task SDK.

Supports sub-commands:
- security: Security scanning and analysis tasks

Examples:
  devops quality security scan
  devops quality security --vars SCAN_TYPE=all scan`,
}

// qualitySecurityCmd represents the quality security command
var qualitySecurityCmd = &cobra.Command{
	Use:   "security [task]",
	Short: "Execute security scanning tasks",
	Long: `Execute security scanning tasks from devops/tasks/quality/security.yaml

Available tasks will be loaded from the Taskfile.

Examples:
  devops quality security           # List available tasks
  devops quality security scan     # Run security scan
  devops quality security report   # Generate security report
  devops quality security audit    # Run security audit`,
	RunE: runQualitySecurity,
}

func runQualitySecurity(cmd *cobra.Command, args []string) error {
	return qualityTaskCmd.RunTask("quality/security.yaml", args, qualityVars, qualityDir)
}

func init() {
	// Initialize task command interface
	qualityTaskCmd = taskcommand.NewBaseTaskCommand()

	// Add quality command to root
	rootCmd.AddCommand(qualityCmd)

	// Add subcommands to quality
	qualityCmd.AddCommand(qualitySecurityCmd)

	// Add flags
	qualityCmd.PersistentFlags().StringVar(&qualityDir, "dir", "", "Project directory (default: current directory)")
	qualityCmd.PersistentFlags().StringToStringVar(&qualityVars, "vars", map[string]string{}, "Task variables (key=value)")
}
