package docker

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os/exec"
	"strings"
	"time"
)

// CLIProvider implements BuildProvider using Docker CLI
type CLIProvider struct {
	logger     *slog.Logger
	binaryPath string
}

// NewCLIProvider creates a new Docker CLI provider
func NewCLIProvider(logger *slog.Logger) *CLIProvider {
	return &CLIProvider{
		logger:     logger.With("provider", "docker-cli"),
		binaryPath: "docker",
	}
}

// Build implements BuildProvider
func (p *CLIProvider) Build(ctx context.Context, options ProviderBuildOptions) error {
	p.logger.InfoContext(ctx, "Building with Docker",
		"tags", options.Tags,
		"dockerfile", options.Dockerfile,
	)

	args := []string{"build"}

	// Add tags
	for _, tag := range options.Tags {
		args = append(args, "-t", tag)
	}

	// Add dockerfile
	if options.Dockerfile != "" {
		args = append(args, "-f", options.Dockerfile)
	}

	// Add build args
	for k, v := range options.BuildArgs {
		args = append(args, "--build-arg", fmt.Sprintf("%s=%s", k, v))
	}

	// Add target
	if options.Target != "" {
		args = append(args, "--target", options.Target)
	}

	// Add platform
	if options.Platform != "" {
		args = append(args, "--platform", options.Platform)
	}

	// Add no-cache
	if options.NoCache {
		args = append(args, "--no-cache")
	}

	// Add context
	args = append(args, options.Context)

	cmd := exec.CommandContext(ctx, p.binaryPath, args...)

	// Set up output stream
	if options.OutputStream != nil {
		cmd.Stdout = options.OutputStream
		cmd.Stderr = options.OutputStream
	}

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("docker build failed: %w", err)
	}

	return nil
}

// Push implements BuildProvider
func (p *CLIProvider) Push(ctx context.Context, image string, auth *AuthConfig) error {
	p.logger.InfoContext(ctx, "Pushing image", "image", image)

	// Login if auth provided
	if auth != nil {
		if err := p.login(ctx, auth); err != nil {
			return fmt.Errorf("docker login failed: %w", err)
		}
		defer func() {
			if err := p.logout(ctx, auth.Server); err != nil {
				p.logger.WarnContext(ctx, "Failed to logout from registry", "error", err)
			}
		}()
	}

	cmd := exec.CommandContext(ctx, p.binaryPath, "push", image)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("docker push failed: %w", err)
	}

	return nil
}

// Tag implements BuildProvider
func (p *CLIProvider) Tag(ctx context.Context, source, target string) error {
	p.logger.InfoContext(ctx, "Tagging image",
		"source", source,
		"target", target,
	)

	cmd := exec.CommandContext(ctx, p.binaryPath, "tag", source, target)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("docker tag failed: %w", err)
	}

	return nil
}

// Pull implements BuildProvider
func (p *CLIProvider) Pull(ctx context.Context, image string, auth *AuthConfig) error {
	p.logger.InfoContext(ctx, "Pulling image", "image", image)

	// Login if auth provided
	if auth != nil {
		if err := p.login(ctx, auth); err != nil {
			return fmt.Errorf("docker login failed: %w", err)
		}
		defer func() {
			if err := p.logout(ctx, auth.Server); err != nil {
				p.logger.WarnContext(ctx, "Failed to logout from registry", "error", err)
			}
		}()
	}

	cmd := exec.CommandContext(ctx, p.binaryPath, "pull", image)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("docker pull failed: %w", err)
	}

	return nil
}

// ImageExists implements BuildProvider
func (p *CLIProvider) ImageExists(ctx context.Context, image string) (bool, error) {
	cmd := exec.CommandContext(ctx, p.binaryPath, "image", "inspect", image)
	if err := cmd.Run(); err != nil {
		// If inspect fails, image doesn't exist
		return false, nil
	}
	return true, nil
}

// RemoveImage implements BuildProvider
func (p *CLIProvider) RemoveImage(ctx context.Context, image string) error {
	p.logger.InfoContext(ctx, "Removing image", "image", image)

	cmd := exec.CommandContext(ctx, p.binaryPath, "rmi", image)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("docker rmi failed: %w", err)
	}

	return nil
}

// ListImages implements BuildProvider
func (p *CLIProvider) ListImages(ctx context.Context) ([]ImageInfo, error) {
	cmd := exec.CommandContext(ctx, p.binaryPath, "images", "--format", "json")
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("docker images failed: %w", err)
	}

	var images []ImageInfo
	lines := strings.Split(string(output), "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		var dockerImage dockerImageJSON
		if err := json.Unmarshal([]byte(line), &dockerImage); err != nil {
			p.logger.WarnContext(ctx, "Failed to parse image JSON",
				"line", line,
				"error", err,
			)
			continue
		}

		// Parse created time
		createdTime, _ := time.Parse(time.RFC3339, dockerImage.CreatedAt)

		// Convert size
		size := p.parseSize(dockerImage.Size)

		img := ImageInfo{
			ID:      dockerImage.ID,
			Tags:    []string{dockerImage.Repository + ":" + dockerImage.Tag},
			Size:    size,
			Created: createdTime.Unix(),
		}

		images = append(images, img)
	}

	return images, nil
}

// login performs docker login
func (p *CLIProvider) login(ctx context.Context, auth *AuthConfig) error {
	args := []string{"login"}

	if auth.Username != "" {
		args = append(args, "-u", auth.Username)
	}

	if auth.Password != "" {
		args = append(args, "-p", auth.Password)
	}

	if auth.Server != "" {
		args = append(args, auth.Server)
	}

	cmd := exec.CommandContext(ctx, p.binaryPath, args...)
	if err := cmd.Run(); err != nil {
		return err
	}

	return nil
}

// logout performs docker logout
func (p *CLIProvider) logout(ctx context.Context, server string) error {
	args := []string{"logout"}

	if server != "" {
		args = append(args, server)
	}

	cmd := exec.CommandContext(ctx, p.binaryPath, args...)
	if err := cmd.Run(); err != nil {
		// Log but don't fail on logout errors
		p.logger.DebugContext(ctx, "Docker logout failed", "error", err)
	}

	return nil
}

// parseSize parses Docker size string to bytes
func (p *CLIProvider) parseSize(sizeStr string) int64 {
	// Simple parsing - in real implementation use proper parsing
	sizeStr = strings.ToUpper(strings.TrimSpace(sizeStr))

	multiplier := int64(1)
	if strings.HasSuffix(sizeStr, "GB") {
		multiplier = 1024 * 1024 * 1024
		sizeStr = strings.TrimSuffix(sizeStr, "GB")
	} else if strings.HasSuffix(sizeStr, "MB") {
		multiplier = 1024 * 1024
		sizeStr = strings.TrimSuffix(sizeStr, "MB")
	} else if strings.HasSuffix(sizeStr, "KB") {
		multiplier = 1024
		sizeStr = strings.TrimSuffix(sizeStr, "KB")
	}

	// Parse number
	var size float64
	if _, err := fmt.Sscanf(sizeStr, "%f", &size); err != nil {
		// Return 0 on parse error
		return 0
	}

	return int64(size * float64(multiplier))
}

// dockerImageJSON represents Docker image JSON format
type dockerImageJSON struct {
	ID         string `json:"ID"`
	Repository string `json:"Repository"`
	Tag        string `json:"Tag"`
	Digest     string `json:"Digest"`
	CreatedAt  string `json:"CreatedAt"`
	Size       string `json:"Size"`
}
