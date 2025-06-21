package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
)

var (
	manifestScenario  string
	manifestOverlay   string
	manifestPath      string
	manifestOutput    string
	manifestDryRun    bool
	manifestValidate  bool
)

// ManifestCmd returns the manifest command
func ManifestCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "manifest",
		Short: "Manage and generate Kubernetes manifests",
		Long: `Generate, validate, and manage Kubernetes manifests from blueprint kustomizations.

This command helps you:
- Generate rendered manifests from kustomize
- Preview what will be deployed
- Validate manifest syntax
- Export manifests for GitOps workflows`,
	}

	// Add subcommands
	cmd.AddCommand(manifestGenerateCmd())
	cmd.AddCommand(manifestRenderCmd())
	cmd.AddCommand(manifestDiffCmd())

	return cmd
}

// manifestGenerateCmd generates manifests from kustomize
func manifestGenerateCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "generate",
		Short: "Generate Kubernetes manifests from blueprint",
		Long: `Generate fully rendered Kubernetes manifests from blueprint kustomizations.

Examples:
  # Generate manifests for minimal scenario
  iaac manifest generate --scenario minimal

  # Generate manifests with local overlay
  iaac manifest generate --scenario full-stack --overlay local

  # Generate and save to file
  iaac manifest generate --scenario development --output manifests.yaml

  # Generate from custom path
  iaac manifest generate --path ./my-kustomization`,
		RunE: runManifestGenerate,
	}

	cmd.Flags().StringVar(&manifestScenario, "scenario", "minimal", "Blueprint scenario to generate")
	cmd.Flags().StringVar(&manifestOverlay, "overlay", "", "Overlay to apply (local, dev)")
	cmd.Flags().StringVar(&manifestPath, "path", "", "Custom kustomization path")
	cmd.Flags().StringVarP(&manifestOutput, "output", "o", "", "Output file (default: stdout)")
	cmd.Flags().BoolVar(&manifestValidate, "validate", true, "Validate generated manifests")

	return cmd
}

// manifestRenderCmd renders manifests with variable substitution
func manifestRenderCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "render",
		Short: "Render manifests with environment substitution",
		Long: `Render Kubernetes manifests with environment variable substitution.

This is useful for:
- Replacing placeholders with actual values
- Preparing manifests for different environments
- GitOps workflows with environment-specific values`,
		RunE: runManifestRender,
	}

	cmd.Flags().StringVar(&manifestPath, "path", "", "Path to manifests or kustomization")
	cmd.Flags().StringVarP(&manifestOutput, "output", "o", "", "Output file (default: stdout)")
	cmd.Flags().BoolVar(&manifestDryRun, "dry-run", false, "Show what would be rendered")

	return cmd
}

// manifestDiffCmd shows differences between current and new manifests
func manifestDiffCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "diff",
		Short: "Show differences between manifests",
		Long: `Compare generated manifests with what's currently deployed in the cluster.

This helps you:
- Preview changes before applying
- Understand what will be updated
- Verify manifest changes`,
		RunE: runManifestDiff,
	}

	cmd.Flags().StringVar(&manifestScenario, "scenario", "minimal", "Blueprint scenario to compare")
	cmd.Flags().StringVar(&manifestOverlay, "overlay", "", "Overlay to apply")
	cmd.Flags().StringVar(&testNamespace, "namespace", "", "Namespace to compare (empty for all)")

	return cmd
}

func runManifestGenerate(cmd *cobra.Command, args []string) error {
	// Determine kustomization path
	kustomizePath := manifestPath
	if kustomizePath == "" {
		// Build path from scenario and overlay
		kustomizePath = buildKustomizationPath(manifestScenario, manifestOverlay)
	}

	fmt.Printf("Generating manifests from: %s\n", kustomizePath)

	// Check if kustomize is available
	if err := checkKustomize(); err != nil {
		return err
	}

	// Run kustomize build
	manifests, err := runKustomizeBuild(kustomizePath)
	if err != nil {
		return fmt.Errorf("failed to generate manifests: %w", err)
	}

	// Validate if requested
	if manifestValidate {
		fmt.Println("Validating generated manifests...")
		if err := validateGeneratedManifests(manifests); err != nil {
			return fmt.Errorf("manifest validation failed: %w", err)
		}
		fmt.Println("✅ Manifests are valid")
	}

	// Output manifests
	if manifestOutput != "" {
		// Write to file
		if err := os.WriteFile(manifestOutput, []byte(manifests), 0644); err != nil {
			return fmt.Errorf("failed to write output file: %w", err)
		}
		fmt.Printf("Manifests written to: %s\n", manifestOutput)
	} else {
		// Print to stdout
		fmt.Println("\n---")
		fmt.Print(manifests)
	}

	return nil
}

func runManifestRender(cmd *cobra.Command, args []string) error {
	if manifestPath == "" {
		return fmt.Errorf("--path is required for render command")
	}

	fmt.Printf("Rendering manifests from: %s\n", manifestPath)

	// Read manifest files
	manifests, err := readManifestFiles(manifestPath)
	if err != nil {
		return fmt.Errorf("failed to read manifests: %w", err)
	}

	// Perform environment substitution
	rendered := performEnvSubstitution(manifests)

	if manifestDryRun {
		fmt.Println("Dry run - showing rendered output:")
		fmt.Println("=====================================")
	}

	// Output rendered manifests
	if manifestOutput != "" && !manifestDryRun {
		if err := os.WriteFile(manifestOutput, []byte(rendered), 0644); err != nil {
			return fmt.Errorf("failed to write output file: %w", err)
		}
		fmt.Printf("Rendered manifests written to: %s\n", manifestOutput)
	} else {
		fmt.Print(rendered)
	}

	return nil
}

func runManifestDiff(cmd *cobra.Command, args []string) error {
	fmt.Println("Comparing manifests with deployed resources...")

	// Generate manifests for comparison
	kustomizePath := buildKustomizationPath(manifestScenario, manifestOverlay)
	_, err := runKustomizeBuild(kustomizePath)
	if err != nil {
		return fmt.Errorf("failed to generate manifests: %w", err)
	}

	// Get current manifests from cluster
	// TODO: Implement actual diff with cluster
	// This would use kubectl diff or similar functionality

	fmt.Println("\n🔍 Manifest differences:")
	fmt.Println("========================")
	fmt.Println("(Diff functionality not yet implemented)")
	fmt.Printf("Would compare %s scenario with cluster state\n", manifestScenario)

	return nil
}

// Helper functions

func checkKustomize() error {
	_, err := exec.LookPath("kustomize")
	if err != nil {
		// Try kubectl kustomize as fallback
		if _, err := exec.LookPath("kubectl"); err != nil {
			return fmt.Errorf("neither kustomize nor kubectl found in PATH")
		}
	}
	return nil
}

func buildKustomizationPath(scenario, overlay string) string {
	// Find iaac path
	iaacPath := findIaacPathForManifest(".")
	if iaacPath == "" {
		// Use relative path as fallback
		iaacPath = "."
	}

	// Build scenario path
	scenarioPath := filepath.Join(iaacPath, "blueprint", "scenarios", scenario)

	// If overlay is specified, create a temporary kustomization that combines both
	if overlay != "" {
		// For now, just return the scenario path
		// TODO: Implement overlay merging
		return scenarioPath
	}

	return scenarioPath
}

func runKustomizeBuild(path string) (string, error) {
	// Try kustomize first
	if _, err := exec.LookPath("kustomize"); err == nil {
		cmd := exec.Command("kustomize", "build", path)
		output, err := cmd.CombinedOutput()
		if err != nil {
			return "", fmt.Errorf("kustomize build failed: %w\n%s", err, string(output))
		}
		return string(output), nil
	}

	// Fall back to kubectl
	cmd := exec.Command("kubectl", "kustomize", path)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("kubectl kustomize failed: %w\n%s", err, string(output))
	}
	return string(output), nil
}

func validateGeneratedManifests(manifests string) error {
	// Create a temporary file
	tmpFile, err := os.CreateTemp("", "manifests-*.yaml")
	if err != nil {
		return err
	}
	defer os.Remove(tmpFile.Name())

	// Write manifests to temp file
	if _, err := tmpFile.WriteString(manifests); err != nil {
		return err
	}
	tmpFile.Close()

	// Run kubectl dry-run
	cmd := exec.Command("kubectl", "apply", "--dry-run=client", "-f", tmpFile.Name())
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("validation failed: %w\n%s", err, string(output))
	}

	return nil
}

func readManifestFiles(path string) (string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}

	var content strings.Builder

	if info.IsDir() {
		// Read all YAML files in directory
		files, err := filepath.Glob(filepath.Join(path, "*.yaml"))
		if err != nil {
			return "", err
		}

		yamlFiles, err := filepath.Glob(filepath.Join(path, "*.yml"))
		if err != nil {
			return "", err
		}
		files = append(files, yamlFiles...)

		for i, file := range files {
			data, err := os.ReadFile(file)
			if err != nil {
				return "", err
			}

			if i > 0 {
				content.WriteString("\n---\n")
			}
			content.Write(data)
		}
	} else {
		// Single file
		data, err := os.ReadFile(path)
		if err != nil {
			return "", err
		}
		content.Write(data)
	}

	return content.String(), nil
}

func performEnvSubstitution(manifests string) string {
	// Simple environment variable substitution
	// Replace ${VAR_NAME} with environment variable values
	
	result := manifests
	
	// Find all ${...} patterns
	for {
		start := strings.Index(result, "${")
		if start == -1 {
			break
		}
		
		end := strings.Index(result[start:], "}")
		if end == -1 {
			break
		}
		end += start
		
		// Extract variable name
		varName := result[start+2 : end]
		
		// Get value from environment
		value := os.Getenv(varName)
		if value == "" {
			// Optionally, you could leave it unchanged or use a default
			value = fmt.Sprintf("${%s}", varName)
		}
		
		// Replace
		result = result[:start] + value + result[end+1:]
	}
	
	return result
}

// findIaacPathForManifest finds the iaac directory by walking up the directory tree
func findIaacPathForManifest(startPath string) string {
	// Convert to absolute path
	absPath, err := filepath.Abs(startPath)
	if err != nil {
		return ""
	}

	// Walk up the directory tree
	current := absPath
	for {
		// Check if iaac/blueprint exists
		blueprintPath := filepath.Join(current, "iaac", "blueprint")
		if info, err := os.Stat(blueprintPath); err == nil && info.IsDir() {
			return current
		}

		// Also check if we're already in iaac directory
		if filepath.Base(current) == "iaac" {
			blueprintPath := filepath.Join(current, "blueprint")
			if info, err := os.Stat(blueprintPath); err == nil && info.IsDir() {
				return filepath.Dir(current)
			}
		}

		// Move up one directory
		parent := filepath.Dir(current)
		if parent == current {
			// Reached root
			break
		}
		current = parent
	}

	return ""
}