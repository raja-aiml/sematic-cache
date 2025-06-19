package docker

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/api/types/network"
	"github.com/docker/docker/client"
	"github.com/docker/docker/pkg/stdcopy"
	"github.com/docker/go-connections/nat"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
)

// SDKBuilder implements Docker operations using the Docker SDK
type SDKBuilder struct {
	client *client.Client
	logger *utils.Logger
}

// NewSDKBuilder creates a new Docker SDK builder
func NewSDKBuilder() (*SDKBuilder, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, fmt.Errorf("failed to create docker client: %w", err)
	}

	return &SDKBuilder{
		client: cli,
		logger: utils.NewLogger("docker-sdk"),
	}, nil
}

// Close closes the Docker client connection
func (b *SDKBuilder) Close() error {
	if b.client != nil {
		return b.client.Close()
	}
	return nil
}

// Build builds a Docker image using the SDK
func (b *SDKBuilder) Build(ctx context.Context, dockerfilePath, imageName, buildContext string) error {
	b.logger.Info("Building image: %s", imageName)

	// Create tar archive of build context
	tar, err := createTarArchive(buildContext, dockerfilePath)
	if err != nil {
		return fmt.Errorf("failed to create build context: %w", err)
	}

	// Prepare build options
	buildOptions := types.ImageBuildOptions{
		Tags:       []string{imageName},
		Dockerfile: filepath.Base(dockerfilePath),
		Remove:     true,
		PullParent: true,
	}

	// Build the image
	resp, err := b.client.ImageBuild(ctx, tar, buildOptions)
	if err != nil {
		return fmt.Errorf("docker build failed: %w", err)
	}
	defer resp.Body.Close()

	// Process build output
	if err := processBuildOutput(resp.Body, b.logger); err != nil {
		return fmt.Errorf("build output processing failed: %w", err)
	}

	b.logger.Info("Image built successfully: %s", imageName)
	return nil
}

// Tag tags a Docker image
func (b *SDKBuilder) Tag(ctx context.Context, sourceImage, targetImage string) error {
	b.logger.Info("Tagging %s as %s", sourceImage, targetImage)

	if err := b.client.ImageTag(ctx, sourceImage, targetImage); err != nil {
		return fmt.Errorf("docker tag failed: %w", err)
	}

	return nil
}

// Push pushes a Docker image to a registry
func (b *SDKBuilder) Push(ctx context.Context, imageName string) error {
	b.logger.Info("Pushing image: %s", imageName)

	// TODO: Add authentication support
	pushOptions := image.PushOptions{}

	resp, err := b.client.ImagePush(ctx, imageName, pushOptions)
	if err != nil {
		return fmt.Errorf("docker push failed: %w", err)
	}
	defer resp.Close()

	// Process push output
	if err := processPushOutput(resp, b.logger); err != nil {
		return fmt.Errorf("push output processing failed: %w", err)
	}

	return nil
}

// Run runs a Docker container
func (b *SDKBuilder) Run(ctx context.Context, imageName string, opts *RunOptions) (string, error) {
	config := &container.Config{
		Image: imageName,
	}

	hostConfig := &container.HostConfig{
		AutoRemove: true,
	}

	// Apply options
	if opts != nil {
		if len(opts.Env) > 0 {
			config.Env = make([]string, 0, len(opts.Env))
			for k, v := range opts.Env {
				config.Env = append(config.Env, fmt.Sprintf("%s=%s", k, v))
			}
		}

		if len(opts.Volumes) > 0 {
			hostConfig.Binds = opts.Volumes
		}

		if len(opts.Ports) > 0 {
			config.ExposedPorts = make(nat.PortSet)
			hostConfig.PortBindings = make(nat.PortMap)
			
			for hostPort, containerPort := range opts.Ports {
				port, err := nat.NewPort("tcp", containerPort)
				if err != nil {
					return "", fmt.Errorf("invalid port: %w", err)
				}
				config.ExposedPorts[port] = struct{}{}
				hostConfig.PortBindings[port] = []nat.PortBinding{
					{HostPort: hostPort},
				}
			}
		}

		if opts.Network != "" {
			hostConfig.NetworkMode = container.NetworkMode(opts.Network)
		}

		if len(opts.Command) > 0 {
			config.Cmd = opts.Command
		}
	}

	// Create container
	containerName := ""
	if opts != nil && opts.Name != "" {
		containerName = opts.Name
	}

	resp, err := b.client.ContainerCreate(ctx, config, hostConfig, &network.NetworkingConfig{}, nil, containerName)
	if err != nil {
		return "", fmt.Errorf("container create failed: %w", err)
	}

	// Start container
	if err := b.client.ContainerStart(ctx, resp.ID, container.StartOptions{}); err != nil {
		return "", fmt.Errorf("container start failed: %w", err)
	}

	// Handle detached mode
	if opts != nil && opts.Detach {
		return resp.ID, nil
	}

	// Wait for container to finish
	statusCh, errCh := b.client.ContainerWait(ctx, resp.ID, container.WaitConditionNotRunning)
	select {
	case err := <-errCh:
		if err != nil {
			return "", fmt.Errorf("container wait failed: %w", err)
		}
	case <-statusCh:
	}

	// Get container logs
	logOptions := container.LogsOptions{
		ShowStdout: true,
		ShowStderr: true,
	}

	logs, err := b.client.ContainerLogs(ctx, resp.ID, logOptions)
	if err != nil {
		return "", fmt.Errorf("failed to get container logs: %w", err)
	}
	defer logs.Close()

	// Read logs
	var buf bytes.Buffer
	if _, err := stdcopy.StdCopy(&buf, &buf, logs); err != nil {
		return "", fmt.Errorf("failed to read logs: %w", err)
	}

	return strings.TrimSpace(buf.String()), nil
}

// Stop stops a running container
func (b *SDKBuilder) Stop(ctx context.Context, containerID string) error {
	timeout := 10 // seconds
	stopOptions := container.StopOptions{
		Timeout: &timeout,
	}

	if err := b.client.ContainerStop(ctx, containerID, stopOptions); err != nil {
		return fmt.Errorf("docker stop failed: %w", err)
	}

	return nil
}

// Remove removes a container
func (b *SDKBuilder) Remove(ctx context.Context, containerID string) error {
	removeOptions := container.RemoveOptions{
		Force: true,
	}

	if err := b.client.ContainerRemove(ctx, containerID, removeOptions); err != nil {
		return fmt.Errorf("docker rm failed: %w", err)
	}

	return nil
}

// IsDockerRunning checks if Docker daemon is running
func (b *SDKBuilder) IsDockerRunning(ctx context.Context) bool {
	_, err := b.client.Ping(ctx)
	return err == nil
}

// ListImages lists Docker images
func (b *SDKBuilder) ListImages(ctx context.Context) ([]image.Summary, error) {
	return b.client.ImageList(ctx, image.ListOptions{})
}

// Helper functions

func createTarArchive(contextPath, dockerfilePath string) (io.Reader, error) {
	buf := new(bytes.Buffer)
	tw := tar.NewWriter(buf)
	defer tw.Close()

	// Walk the build context and add files to tar
	err := filepath.Walk(contextPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip .git and other unwanted files
		if strings.Contains(path, ".git") || strings.Contains(path, "node_modules") {
			return filepath.SkipDir
		}

		// Create tar header
		header, err := tar.FileInfoHeader(info, info.Name())
		if err != nil {
			return err
		}

		// Update header name to be relative to context
		relPath, err := filepath.Rel(contextPath, path)
		if err != nil {
			return err
		}
		header.Name = relPath

		// Write header
		if err := tw.WriteHeader(header); err != nil {
			return err
		}

		// If not a directory, write file content
		if !info.IsDir() {
			file, err := os.Open(path)
			if err != nil {
				return err
			}
			defer file.Close()

			if _, err := io.Copy(tw, file); err != nil {
				return err
			}
		}

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to create tar archive: %w", err)
	}

	return buf, nil
}

func processBuildOutput(reader io.Reader, logger *utils.Logger) error {
	decoder := json.NewDecoder(reader)
	for {
		var message map[string]interface{}
		if err := decoder.Decode(&message); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		if stream, ok := message["stream"].(string); ok {
			logger.Debug(strings.TrimSpace(stream))
		}

		if errorDetail, ok := message["errorDetail"].(map[string]interface{}); ok {
			if msg, ok := errorDetail["message"].(string); ok {
				return fmt.Errorf("build error: %s", msg)
			}
		}
	}
	return nil
}

func processPushOutput(reader io.Reader, logger *utils.Logger) error {
	decoder := json.NewDecoder(reader)
	for {
		var message map[string]interface{}
		if err := decoder.Decode(&message); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		if status, ok := message["status"].(string); ok {
			logger.Debug(status)
		}

		if errorDetail, ok := message["errorDetail"].(map[string]interface{}); ok {
			if msg, ok := errorDetail["message"].(string); ok {
				return fmt.Errorf("push error: %s", msg)
			}
		}
	}
	return nil
}