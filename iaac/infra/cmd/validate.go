package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/validation"
	"github.com/spf13/cobra"
)

var (
	validatePath      string
	validateStrict    bool
	validateRecursive bool
	validateFormat    string
)

// ValidateCmd returns the validate command
func ValidateCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "validate [blueprint|manifests|deployment]",
		Short: "Validate configurations and deployments",
		Long: `Validate blueprint structure, Kubernetes manifests, or deployed resources.

Available validation targets:
  blueprint   - Validate blueprint directory structure and kustomization files
  manifests   - Validate Kubernetes manifest files (YAML)
  deployment  - Validate deployed resources in the cluster

Examples:
  # Validate blueprint structure
  iaac validate blueprint --path ./iaac/blueprint

  # Validate all manifests in a directory
  iaac validate manifests --path ./manifests --recursive

  # Validate current deployment
  iaac validate deployment --namespace app`,
	}

	// Add subcommands
	cmd.AddCommand(validateBlueprintCmd())
	cmd.AddCommand(validateManifestsCmd())
	cmd.AddCommand(validateDeploymentCmd())

	return cmd
}

// validateBlueprintCmd validates blueprint structure
func validateBlueprintCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "blueprint",
		Short: "Validate blueprint directory structure",
		Long: `Validate that the blueprint follows the expected structure with proper
kustomization files, scenarios, modules, and overlays.`,
		RunE: runValidateBlueprint,
	}

	cmd.Flags().StringVar(&validatePath, "path", "./iaac/blueprint", "Path to blueprint directory")
	cmd.Flags().BoolVar(&validateStrict, "strict", false, "Enable strict validation")

	return cmd
}

// validateManifestsCmd validates Kubernetes manifests
func validateManifestsCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "manifests",
		Short: "Validate Kubernetes manifest files",
		Long: `Validate YAML files contain valid Kubernetes resources with proper
apiVersion, kind, and required fields.`,
		RunE: runValidateManifests,
	}

	cmd.Flags().StringVar(&validatePath, "path", ".", "Path to manifest files or directory")
	cmd.Flags().BoolVar(&validateRecursive, "recursive", false, "Validate files recursively")
	cmd.Flags().StringVar(&validateFormat, "format", "text", "Output format (text, json)")

	return cmd
}

// validateDeploymentCmd validates deployed resources
func validateDeploymentCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "deployment",
		Short: "Validate deployed resources in cluster",
		Long: `Validate that deployed resources match expectations, including
health checks, resource quotas, and configuration.`,
		RunE: runValidateDeployment,
	}

	cmd.Flags().StringVar(&testNamespace, "namespace", "", "Namespace to validate (empty for all)")
	cmd.Flags().StringVar(&testScenario, "scenario", "minimal", "Expected scenario deployment")

	return cmd
}

func runValidateBlueprint(cmd *cobra.Command, args []string) error {
	fmt.Printf("Validating blueprint at: %s\n", validatePath)

	// Check if path exists
	if _, err := os.Stat(validatePath); os.IsNotExist(err) {
		return fmt.Errorf("blueprint path does not exist: %s", validatePath)
	}

	validator := validation.NewBlueprintValidator(validateStrict)
	results, err := validator.Validate(validatePath)
	if err != nil {
		return fmt.Errorf("validation failed: %w", err)
	}

	// Print results
	printBlueprintValidationResults(results)

	if !results.IsValid() {
		return fmt.Errorf("blueprint validation failed with %d errors", len(results.Errors))
	}

	fmt.Println("\n✅ Blueprint validation passed!")
	return nil
}

func runValidateManifests(cmd *cobra.Command, args []string) error {
	fmt.Printf("Validating manifests at: %s\n", validatePath)

	// Get list of files to validate
	files, err := getManifestFiles(validatePath, validateRecursive)
	if err != nil {
		return fmt.Errorf("failed to find manifest files: %w", err)
	}

	if len(files) == 0 {
		return fmt.Errorf("no manifest files found at %s", validatePath)
	}

	fmt.Printf("Found %d manifest files to validate\n\n", len(files))

	validator := validation.NewManifestValidator()
	allResults := make(map[string]*validation.ValidationResult)
	hasErrors := false

	for _, file := range files {
		result, err := validator.ValidateFile(file)
		if err != nil {
			return fmt.Errorf("failed to validate %s: %w", file, err)
		}

		allResults[file] = result
		if !result.IsValid() {
			hasErrors = true
		}
	}

	// Print results based on format
	switch validateFormat {
	case "json":
		printManifestResultsJSON(allResults)
	default:
		printManifestResultsText(allResults)
	}

	if hasErrors {
		return fmt.Errorf("manifest validation failed")
	}

	fmt.Println("\n✅ All manifests are valid!")
	return nil
}

func runValidateDeployment(cmd *cobra.Command, args []string) error {
	fmt.Println("Validating deployment...")

	// TODO: Initialize kubernetes client
	// client := kubernetes.NewClient()

	validator := validation.NewDeploymentValidator(nil) // Pass actual client

	opts := validation.DeploymentValidationOptions{
		Namespace: testNamespace,
		Scenario:  testScenario,
	}

	results, err := validator.Validate(opts)
	if err != nil {
		return fmt.Errorf("deployment validation failed: %w", err)
	}

	printDeploymentValidationResults(results)

	if !results.IsValid() {
		return fmt.Errorf("deployment validation failed with %d errors", len(results.Errors))
	}

	fmt.Println("\n✅ Deployment validation passed!")
	return nil
}

// Helper functions

func getManifestFiles(path string, recursive bool) ([]string, error) {
	var files []string

	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}

	if info.IsDir() {
		if recursive {
			err = filepath.Walk(path, func(filePath string, info os.FileInfo, err error) error {
				if err != nil {
					return err
				}
				if !info.IsDir() && (strings.HasSuffix(filePath, ".yaml") || strings.HasSuffix(filePath, ".yml")) {
					files = append(files, filePath)
				}
				return nil
			})
		} else {
			entries, err := os.ReadDir(path)
			if err != nil {
				return nil, err
			}
			for _, entry := range entries {
				if !entry.IsDir() && (strings.HasSuffix(entry.Name(), ".yaml") || strings.HasSuffix(entry.Name(), ".yml")) {
					files = append(files, filepath.Join(path, entry.Name()))
				}
			}
		}
	} else {
		// Single file
		if strings.HasSuffix(path, ".yaml") || strings.HasSuffix(path, ".yml") {
			files = append(files, path)
		}
	}

	return files, err
}

func printBlueprintValidationResults(results *validation.ValidationResult) {
	if len(results.Errors) > 0 {
		fmt.Println("❌ Errors:")
		for _, err := range results.Errors {
			fmt.Printf("  - %s\n", err)
		}
		fmt.Println()
	}

	if len(results.Warnings) > 0 {
		fmt.Println("⚠️  Warnings:")
		for _, warn := range results.Warnings {
			fmt.Printf("  - %s\n", warn)
		}
		fmt.Println()
	}

	if len(results.Info) > 0 {
		fmt.Println("ℹ️  Info:")
		for _, info := range results.Info {
			fmt.Printf("  - %s\n", info)
		}
	}
}

func printManifestResultsText(results map[string]*validation.ValidationResult) {
	for file, result := range results {
		relPath, _ := filepath.Rel(".", file)

		if result.IsValid() {
			fmt.Printf("✅ %s\n", relPath)
		} else {
			fmt.Printf("❌ %s\n", relPath)
			for _, err := range result.Errors {
				fmt.Printf("   - %s\n", err)
			}
		}

		if len(result.Warnings) > 0 {
			for _, warn := range result.Warnings {
				fmt.Printf("   ⚠️  %s\n", warn)
			}
		}
	}
}

func printManifestResultsJSON(results map[string]*validation.ValidationResult) {
	// TODO: Implement JSON output
	fmt.Println("{}")
}

func printDeploymentValidationResults(results *validation.ValidationResult) {
	fmt.Println("\nDeployment Validation Results:")
	fmt.Println("=============================")

	// Print checks performed
	if details, ok := results.Details["checks"].([]string); ok {
		fmt.Println("\nChecks performed:")
		for _, check := range details {
			fmt.Printf("  ✓ %s\n", check)
		}
	}

	printBlueprintValidationResults(results)
}
