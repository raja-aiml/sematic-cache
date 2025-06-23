package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

var validateCmd = &cobra.Command{
	Use:   "validate [files...]",
	Short: "Validate Taskfile syntax",
	Long: `Validate the syntax of Taskfile.yaml files.

If no files are specified, it will search for all Taskfile*.yaml and Taskfile*.yml
files in the current directory and subdirectories.`,
	Example: `  # Validate all Taskfiles in current directory
  devops validate

  # Validate specific files
  devops validate Taskfile.yaml iaac/Taskfile.yaml

  # Validate with verbose output
  devops validate -v`,
	RunE: runValidate,
}

var validateVerbose bool

func init() {
	validateCmd.Flags().BoolVarP(&validateVerbose, "verbose", "v", false, "Show verbose output")
}

func runValidate(cmd *cobra.Command, args []string) error {
	var files []string

	if len(args) > 0 {
		files = args
	} else {
		// Find all Taskfiles
		err := filepath.Walk(".", func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return nil
			}
			if strings.Contains(path, ".git") {
				return nil
			}
			if strings.HasPrefix(info.Name(), "Taskfile") &&
				(strings.HasSuffix(info.Name(), ".yaml") || strings.HasSuffix(info.Name(), ".yml")) {
				files = append(files, path)
			}
			return nil
		})
		if err != nil {
			return fmt.Errorf("failed to walk directory: %w", err)
		}
	}

	if len(files) == 0 {
		return fmt.Errorf("no Taskfiles found")
	}

	var hasErrors bool
	validCount := 0

	for _, file := range files {
		if validateVerbose {
			fmt.Printf("Checking %s... ", file)
		}

		data, err := os.ReadFile(file)
		if err != nil {
			fmt.Printf("❌ %s: %v\n", file, err)
			hasErrors = true
			continue
		}

		var content map[string]interface{}
		if err := yaml.Unmarshal(data, &content); err != nil {
			fmt.Printf("❌ %s: %v\n", file, err)
			hasErrors = true
			continue
		}

		// Basic validation
		if _, ok := content["version"]; !ok {
			fmt.Printf("⚠️  %s: missing 'version' field\n", file)
		}

		if validateVerbose {
			fmt.Printf("✅\n")
		}
		validCount++
	}

	if !validateVerbose && !hasErrors {
		fmt.Printf("✅ All %d Taskfiles are valid\n", validCount)
	} else if hasErrors {
		return fmt.Errorf("validation failed")
	}

	return nil
}

