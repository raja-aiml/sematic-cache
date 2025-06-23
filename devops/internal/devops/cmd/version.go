package cmd

import (
	"fmt"
	"runtime"

	"github.com/spf13/cobra"
)

// Note: Version variables are defined in root.go

var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print version information",
	Long:  `Print detailed version information about the devops CLI tool.`,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Fprintf(cmd.OutOrStdout(), "DevOps CLI Tool\n")
		fmt.Fprintf(cmd.OutOrStdout(), "Version:    %s\n", Version)
		fmt.Fprintf(cmd.OutOrStdout(), "Git Commit: %s\n", Commit)
		fmt.Fprintf(cmd.OutOrStdout(), "Build Time: %s\n", Date)
		fmt.Fprintf(cmd.OutOrStdout(), "Built By:   %s\n", BuiltBy)
		fmt.Fprintf(cmd.OutOrStdout(), "Go Version: %s\n", GoVersion)
		fmt.Fprintf(cmd.OutOrStdout(), "Platform:   %s/%s\n", runtime.GOOS, runtime.GOARCH)
	},
}
