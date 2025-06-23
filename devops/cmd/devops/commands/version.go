// Package commands provides the version command
package commands

import (
	"fmt"
	"runtime"

	"github.com/spf13/cobra"

	"github.com/raja-aiml/sematic-cache/devops/pkg/factory"
)

// Version information (set by main.go)
var (
	version   = "dev"
	commit    = "none"
	date      = "unknown"
	builtBy   = "unknown"
	goVersion = runtime.Version()
)

// SetVersionInfo sets the version information
func SetVersionInfo(v, c, d, b, g string) {
	version = v
	commit = c
	date = d
	builtBy = b
	if g != "" {
		goVersion = g
	}
}

// VersionCommand handles version display
type VersionCommand struct {
	*BaseCommand
	cmd *cobra.Command
}

// NewVersionCommand creates a new version command
func NewVersionCommand(factory *factory.Factory) *VersionCommand {
	vc := &VersionCommand{
		BaseCommand: NewBaseCommand(factory),
	}

	vc.cmd = &cobra.Command{
		Use:   "version",
		Short: "Display version information",
		Long:  "Display detailed version information about the devops tool",
		RunE:  vc.run,
	}

	return vc
}

// GetCommand returns the cobra command
func (vc *VersionCommand) GetCommand() *cobra.Command {
	return vc.cmd
}

// run executes the version command
func (vc *VersionCommand) run(cmd *cobra.Command, args []string) error {
	fmt.Printf("DevOps CLI Tool\n")
	fmt.Printf("Version:    %s\n", version)
	fmt.Printf("Commit:     %s\n", commit)
	fmt.Printf("Built:      %s\n", date)
	fmt.Printf("Built By:   %s\n", builtBy)
	fmt.Printf("Go Version: %s\n", goVersion)
	fmt.Printf("Platform:   %s/%s\n", runtime.GOOS, runtime.GOARCH)

	return nil
}
