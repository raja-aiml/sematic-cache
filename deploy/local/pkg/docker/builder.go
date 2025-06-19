package docker

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
)

type Builder struct {
	logger     *utils.Logger
	sdkBuilder *SDKBuilder
	useSDK     bool
}

func NewBuilder() *Builder {
	builder := &Builder{
		logger: utils.NewLogger("docker"),
		useSDK: true,
	}

	// Try to create SDK builder
	if sdkBuilder, err := NewSDKBuilder(); err == nil {
		builder.sdkBuilder = sdkBuilder
		builder.logger.Info("Using Docker SDK")
	} else {
		builder.useSDK = false
		builder.logger.Info("Falling back to Docker CLI: %v", err)
	}

	return builder
}

func (b *Builder) Build(ctx context.Context, dockerfilePath, imageName, buildContext string) error {
	// Use SDK if available
	if b.useSDK && b.sdkBuilder != nil {
		return b.sdkBuilder.Build(ctx, dockerfilePath, imageName, buildContext)
	}

	// Fallback to CLI
	b.logger.Info("Building image: %s", imageName)

	args := []string{
		"build",
		"-f", dockerfilePath,
		"-t", imageName,
		buildContext,
	}

	output, err := utils.RunCommand(ctx, "docker", args, nil)
	if err != nil {
		return fmt.Errorf("docker build failed: %w", err)
	}

	b.logger.Debug("Build output: %s", output)
	b.logger.Info("Image built successfully: %s", imageName)
	return nil
}

func (b *Builder) Tag(ctx context.Context, sourceImage, targetImage string) error {
	// Use SDK if available
	if b.useSDK && b.sdkBuilder != nil {
		return b.sdkBuilder.Tag(ctx, sourceImage, targetImage)
	}

	// Fallback to CLI
	b.logger.Info("Tagging %s as %s", sourceImage, targetImage)

	args := []string{"tag", sourceImage, targetImage}

	_, err := utils.RunCommand(ctx, "docker", args, nil)
	if err != nil {
		return fmt.Errorf("docker tag failed: %w", err)
	}

	return nil
}

func (b *Builder) Push(ctx context.Context, imageName string) error {
	// Use SDK if available
	if b.useSDK && b.sdkBuilder != nil {
		return b.sdkBuilder.Push(ctx, imageName)
	}

	// Fallback to CLI
	b.logger.Info("Pushing image: %s", imageName)

	args := []string{"push", imageName}

	output, err := utils.RunCommand(ctx, "docker", args, nil)
	if err != nil {
		return fmt.Errorf("docker push failed: %w", err)
	}

	b.logger.Debug("Push output: %s", output)
	return nil
}

func (b *Builder) ImportToK3d(ctx context.Context, imageName, clusterName string) error {
	b.logger.Info("Importing image %s to k3d cluster %s", imageName, clusterName)

	args := []string{"image", "import", imageName, "-c", clusterName}

	output, err := utils.RunCommand(ctx, "k3d", args, nil)
	if err != nil {
		return fmt.Errorf("k3d image import failed: %w", err)
	}

	b.logger.Debug("Import output: %s", output)
	b.logger.Info("Image imported successfully")
	return nil
}

func (b *Builder) Run(ctx context.Context, imageName string, opts *RunOptions) (string, error) {
	// Use SDK if available
	if b.useSDK && b.sdkBuilder != nil {
		return b.sdkBuilder.Run(ctx, imageName, opts)
	}

	// Fallback to CLI
	args := []string{"run", "--rm"}

	if opts != nil {
		if opts.Name != "" {
			args = append(args, "--name", opts.Name)
		}
		for k, v := range opts.Env {
			args = append(args, "-e", fmt.Sprintf("%s=%s", k, v))
		}
		for _, v := range opts.Volumes {
			args = append(args, "-v", v)
		}
		for k, v := range opts.Ports {
			args = append(args, "-p", fmt.Sprintf("%s:%s", k, v))
		}
		if opts.Network != "" {
			args = append(args, "--network", opts.Network)
		}
		if opts.Detach {
			args = append(args, "-d")
		}
	}

	args = append(args, imageName)
	if opts != nil && len(opts.Command) > 0 {
		args = append(args, opts.Command...)
	}

	output, err := utils.RunCommand(ctx, "docker", args, nil)
	if err != nil {
		return "", fmt.Errorf("docker run failed: %w", err)
	}

	return strings.TrimSpace(output), nil
}

func (b *Builder) Stop(ctx context.Context, containerID string) error {
	// Use SDK if available
	if b.useSDK && b.sdkBuilder != nil {
		return b.sdkBuilder.Stop(ctx, containerID)
	}

	// Fallback to CLI
	args := []string{"stop", containerID}

	_, err := utils.RunCommand(ctx, "docker", args, nil)
	if err != nil {
		return fmt.Errorf("docker stop failed: %w", err)
	}

	return nil
}

func (b *Builder) Remove(ctx context.Context, containerID string) error {
	// Use SDK if available
	if b.useSDK && b.sdkBuilder != nil {
		return b.sdkBuilder.Remove(ctx, containerID)
	}

	// Fallback to CLI
	args := []string{"rm", "-f", containerID}

	_, err := utils.RunCommand(ctx, "docker", args, nil)
	if err != nil {
		return fmt.Errorf("docker rm failed: %w", err)
	}

	return nil
}

type RunOptions struct {
	Name    string
	Env     map[string]string
	Volumes []string
	Ports   map[string]string
	Network string
	Command []string
	Detach  bool
}

// IsDockerRunning checks if Docker daemon is running
func (b *Builder) IsDockerRunning(ctx context.Context) bool {
	// Use SDK if available
	if b.useSDK && b.sdkBuilder != nil {
		return b.sdkBuilder.IsDockerRunning(ctx)
	}

	// Fallback to CLI
	_, err := utils.RunCommand(ctx, "docker", []string{"info"}, &utils.ExecOptions{Silent: true})
	return err == nil
}

// Close closes the Docker client connection if using SDK
func (b *Builder) Close() error {
	if b.sdkBuilder != nil {
		return b.sdkBuilder.Close()
	}
	return nil
}

func GetProjectRoot() (string, error) {
	// Try to find project root by looking for go.mod
	currentDir := "."
	for i := 0; i < 5; i++ {
		modPath := filepath.Join(currentDir, "go.mod")
		if _, err := os.Stat(modPath); err == nil {
			return filepath.Abs(currentDir)
		}
		currentDir = filepath.Join(currentDir, "..")
	}
	return "", fmt.Errorf("could not find project root")
}
