package kubernetes

import (
	"context"
	"fmt"
	"os"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
)

func ApplyKustomize(ctx context.Context, path string, namespace string) error {
	logger := utils.NewLogger("kustomize")

	// Try SDK-based approach first
	if config, err := NewApplyConfig(""); err == nil {
		logger.Info("Using SDK-based kustomize apply")
		return config.ApplyKustomizeSDK(ctx, path, namespace)
	}

	// Fallback to CLI-based approach
	logger.Info("Falling back to CLI-based kustomize apply")
	return applyKustomizeCLI(ctx, path, namespace)
}

func applyKustomizeCLI(ctx context.Context, path string, namespace string) error {
	logger := utils.NewLogger("kustomize")

	// Check if kustomize or kubectl exists
	useKubectl := utils.CommandExists("kubectl")
	useKustomize := utils.CommandExists("kustomize")

	if !useKubectl && !useKustomize {
		return fmt.Errorf("neither kubectl nor kustomize found in PATH")
	}

	var args []string
	var cmd string

	if useKubectl {
		cmd = "kubectl"
		args = []string{"apply", "-k", path}
		if namespace != "" {
			args = append(args, "-n", namespace)
		}
	} else {
		// Use kustomize build | kubectl apply
		kustomizeOut, err := utils.RunCommand(ctx, "kustomize", []string{"build", path}, nil)
		if err != nil {
			return fmt.Errorf("kustomize build failed: %w", err)
		}

		// Write to temp file and apply
		tmpfile, err := os.CreateTemp("", "kustomize-*.yaml")
		if err != nil {
			return fmt.Errorf("failed to create temp file: %w", err)
		}
		defer os.Remove(tmpfile.Name())

		if _, err := tmpfile.WriteString(kustomizeOut); err != nil {
			return fmt.Errorf("failed to write kustomize output: %w", err)
		}
		tmpfile.Close()

		cmd = "kubectl"
		args = []string{"apply", "-f", tmpfile.Name()}
		if namespace != "" {
			args = append(args, "-n", namespace)
		}
	}

	logger.Info("Applying resources from %s", path)

	output, err := utils.RunCommand(ctx, cmd, args, nil)
	if err != nil {
		return fmt.Errorf("failed to apply resources: %w", err)
	}

	logger.Debug("Applied: %s", output)
	return nil
}

func DeleteKustomize(ctx context.Context, path string, namespace string) error {
	logger := utils.NewLogger("kustomize")

	// Try SDK-based approach first
	if config, err := NewApplyConfig(""); err == nil {
		logger.Info("Using SDK-based kustomize delete")
		return config.DeleteKustomizeSDK(ctx, path, namespace)
	}

	// Fallback to CLI-based approach
	logger.Info("Falling back to CLI-based kustomize delete")
	return deleteKustomizeCLI(ctx, path, namespace)
}

func deleteKustomizeCLI(ctx context.Context, path string, namespace string) error {
	logger := utils.NewLogger("kustomize")

	var args []string
	cmd := "kubectl"

	if utils.CommandExists("kubectl") {
		args = []string{"delete", "-k", path}
		if namespace != "" {
			args = append(args, "-n", namespace)
		}
	} else {
		return fmt.Errorf("kubectl not found in PATH")
	}

	logger.Info("Deleting resources from %s", path)

	output, err := utils.RunCommand(ctx, cmd, args, nil)
	if err != nil {
		// Ignore errors if resources don't exist
		logger.Warn("Delete completed with warnings: %v", err)
	} else {
		logger.Debug("Deleted: %s", output)
	}

	return nil
}
