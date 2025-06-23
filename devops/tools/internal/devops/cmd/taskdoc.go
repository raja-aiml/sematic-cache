package cmd

import (
	"fmt"
	"os"

	"github.com/raja-aiml/sematic-cache/devops/tools/internal/taskdoc"
	"github.com/spf13/cobra"
)

var (
	output   string
	format   string
	rootDir  string
	verbose  bool
	flowOnly bool
)

var taskdocCmd = &cobra.Command{
	Use:   "taskdoc",
	Short: "Generate documentation for Taskfiles",
	Long: `Generate comprehensive documentation for all Taskfiles in the project.

This command scans for Taskfile.yaml files and generates documentation
including task hierarchy, dependencies, and usage examples.`,
	Example: `  # Generate markdown documentation
  devops taskdoc

  # Generate JSON documentation
  devops taskdoc --format json --output taskfile-docs.json

  # Generate only task flow
  devops taskdoc --flow

  # Generate documentation for specific directory
  devops taskdoc --root ./iaac`,
	RunE: runTaskdoc,
}

func init() {
	taskdocCmd.Flags().StringVarP(&output, "output", "o", "taskfile-docs.md", "Output file path (use '-' for stdout)")
	taskdocCmd.Flags().StringVarP(&format, "format", "f", "markdown", "Output format (markdown, json)")
	taskdocCmd.Flags().StringVarP(&rootDir, "root", "r", ".", "Root directory to search for Taskfiles")
	taskdocCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "Enable verbose output")
	taskdocCmd.Flags().BoolVar(&flowOnly, "flow", false, "Generate only task flow diagram")
}

func runTaskdoc(cmd *cobra.Command, args []string) error {
	// Create generator with options
	opts := []taskdoc.Option{
		taskdoc.WithRootDir(rootDir),
		taskdoc.WithVerbose(verbose),
	}

	generator, err := taskdoc.NewGenerator(opts...)
	if err != nil {
		return fmt.Errorf("failed to create generator: %w", err)
	}

	// Generate documentation
	var content string
	if flowOnly {
		content, err = generator.GenerateFlow()
	} else {
		switch format {
		case "markdown":
			content, err = generator.GenerateMarkdown()
		case "json":
			content, err = generator.GenerateJSON()
		default:
			return fmt.Errorf("unknown format: %s", format)
		}
	}

	if err != nil {
		return fmt.Errorf("failed to generate documentation: %w", err)
	}

	// Write output
	if output == "-" {
		fmt.Print(content)
	} else {
		if err := os.WriteFile(output, []byte(content), 0644); err != nil {
			return fmt.Errorf("failed to write output file: %w", err)
		}
		fmt.Printf("📄 Documentation generated: %s\n", output)
	}

	return nil
}

