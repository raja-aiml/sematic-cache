// Package docker provides Docker operations using the Docker SDK
package docker

import (
	"context"
	"fmt"
	"io"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/client"
	"github.com/docker/docker/pkg/stdcopy"

	"github.com/raja-aiml/sematic-cache/devops/internal/logger"
)

// Client provides Docker operations using the SDK
type Client struct {
	client *client.Client
	logger *logger.Logger
}

// NewClient creates a new Docker client
func NewClient() (*Client, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, fmt.Errorf("failed to create docker client: %w", err)
	}

	return &Client{
		client: cli,
		logger: logger.New(),
	}, nil
}

// NewClientWithLogger creates a new Docker client with a custom logger
func NewClientWithLogger(l *logger.Logger) (*Client, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, fmt.Errorf("failed to create docker client: %w", err)
	}

	return &Client{
		client: cli,
		logger: l,
	}, nil
}

// Close closes the Docker client connection
func (c *Client) Close() error {
	return c.client.Close()
}

// IsRunning checks if Docker daemon is running
func (c *Client) IsRunning(ctx context.Context) bool {
	_, err := c.client.Ping(ctx)
	return err == nil
}

// Version returns Docker version information
func (c *Client) Version(ctx context.Context) (types.Version, error) {
	return c.client.ServerVersion(ctx)
}

// ContainerInfo holds container information
type ContainerInfo struct {
	ID      string
	Name    string
	Image   string
	Status  string
	State   string
	Health  string
	Created time.Time
}

// ListContainers lists all containers
func (c *Client) ListContainers(ctx context.Context, all bool) ([]ContainerInfo, error) {
	containers, err := c.client.ContainerList(ctx, container.ListOptions{
		All: all,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list containers: %w", err)
	}

	var result []ContainerInfo
	for _, cont := range containers {
		info := ContainerInfo{
			ID:      cont.ID[:12],
			Image:   cont.Image,
			Status:  cont.Status,
			State:   cont.State,
			Created: time.Unix(cont.Created, 0),
		}

		// Get container name
		if len(cont.Names) > 0 {
			info.Name = cont.Names[0]
		}

		result = append(result, info)
	}

	return result, nil
}

// WaitForContainer waits for a container to become healthy
func (c *Client) WaitForContainer(ctx context.Context, containerID string, timeout time.Duration) error {
	c.logger.Info("Waiting for container %s to be healthy...", containerID)

	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		inspect, err := c.client.ContainerInspect(ctx, containerID)
		if err != nil {
			return fmt.Errorf("failed to inspect container: %w", err)
		}

		// Check if container has health check
		if inspect.State.Health != nil {
			switch inspect.State.Health.Status {
			case "healthy":
				c.logger.Success("Container %s is healthy", containerID)
				return nil
			case "unhealthy":
				return fmt.Errorf("container %s is unhealthy", containerID)
			}
		} else if inspect.State.Running {
			// No health check defined, just check if running
			c.logger.Success("Container %s is running", containerID)
			return nil
		}

		select {
		case <-time.After(time.Second):
			// Continue checking
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return fmt.Errorf("container %s failed to become healthy within %v", containerID, timeout)
}

// ExecCommand executes a command in a container
func (c *Client) ExecCommand(ctx context.Context, containerID string, cmd []string) (string, error) {
	execConfig := container.ExecOptions{
		Cmd:          cmd,
		AttachStdout: true,
		AttachStderr: true,
	}

	execID, err := c.client.ContainerExecCreate(ctx, containerID, execConfig)
	if err != nil {
		return "", fmt.Errorf("failed to create exec: %w", err)
	}

	resp, err := c.client.ContainerExecAttach(ctx, execID.ID, container.ExecAttachOptions{})
	if err != nil {
		return "", fmt.Errorf("failed to attach to exec: %w", err)
	}
	defer resp.Close()

	// Read the output
	var output []byte
	output, err = io.ReadAll(resp.Reader)
	if err != nil {
		return "", fmt.Errorf("failed to read exec output: %w", err)
	}

	return string(output), nil
}

// GetContainerLogs retrieves container logs
func (c *Client) GetContainerLogs(ctx context.Context, containerID string, tail string, follow bool) (io.ReadCloser, error) {
	options := container.LogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Tail:       tail,
		Follow:     follow,
	}

	logs, err := c.client.ContainerLogs(ctx, containerID, options)
	if err != nil {
		return nil, fmt.Errorf("failed to get container logs: %w", err)
	}

	return logs, nil
}

// StreamContainerLogs streams container logs to the logger
func (c *Client) StreamContainerLogs(ctx context.Context, containerID string, tail string) error {
	logs, err := c.GetContainerLogs(ctx, containerID, tail, true)
	if err != nil {
		return err
	}
	defer logs.Close()

	// Use stdcopy to demultiplex stdout/stderr
	_, err = stdcopy.StdCopy(
		logger.NewLogWriter(c.logger, logger.InfoLevel),
		logger.NewLogWriter(c.logger, logger.ErrorLevel),
		logs,
	)

	return err
}

// StopContainer stops a running container
func (c *Client) StopContainer(ctx context.Context, containerID string, timeout *time.Duration) error {
	c.logger.Info("Stopping container %s...", containerID)

	var stopTimeout int
	if timeout != nil {
		stopTimeout = int(timeout.Seconds())
	}

	err := c.client.ContainerStop(ctx, containerID, container.StopOptions{
		Timeout: &stopTimeout,
	})
	if err != nil {
		return fmt.Errorf("failed to stop container: %w", err)
	}

	c.logger.Success("Container %s stopped", containerID)
	return nil
}

// RemoveContainer removes a container
func (c *Client) RemoveContainer(ctx context.Context, containerID string, force bool) error {
	c.logger.Info("Removing container %s...", containerID)

	err := c.client.ContainerRemove(ctx, containerID, container.RemoveOptions{
		Force: force,
	})
	if err != nil {
		return fmt.Errorf("failed to remove container: %w", err)
	}

	c.logger.Success("Container %s removed", containerID)
	return nil
}

// ImageInfo holds image information
type ImageInfo struct {
	ID      string
	Tags    []string
	Size    int64
	Created time.Time
}

// ListImages lists Docker images
func (c *Client) ListImages(ctx context.Context) ([]ImageInfo, error) {
	images, err := c.client.ImageList(ctx, image.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to list images: %w", err)
	}

	var result []ImageInfo
	for _, img := range images {
		info := ImageInfo{
			ID:      img.ID,
			Tags:    img.RepoTags,
			Size:    img.Size,
			Created: time.Unix(img.Created, 0),
		}
		result = append(result, info)
	}

	return result, nil
}

// PullImage pulls a Docker image
func (c *Client) PullImage(ctx context.Context, imageName string) error {
	c.logger.Info("Pulling image %s...", imageName)

	reader, err := c.client.ImagePull(ctx, imageName, image.PullOptions{})
	if err != nil {
		return fmt.Errorf("failed to pull image: %w", err)
	}
	defer reader.Close()

	// Read the output to ensure the pull completes
	_, err = io.Copy(io.Discard, reader)
	if err != nil {
		return fmt.Errorf("failed to complete image pull: %w", err)
	}

	c.logger.Success("Image %s pulled successfully", imageName)
	return nil
}
