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
	"github.com/docker/docker/api/types/network"
	"github.com/docker/docker/client"
	"github.com/docker/go-connections/nat"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
)

// Client implements the DockerClient interface
type Client struct {
	client client.APIClient
	logger interfaces.Logger
}

// NewClient creates a new Docker client
func NewClient(logger interfaces.Logger) (interfaces.DockerClient, error) {
	cli, err := client.NewClientWithOpts(
		client.FromEnv,
		client.WithAPIVersionNegotiation(),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create docker client: %w", err)
	}

	return &Client{
		client: cli,
		logger: logger,
	}, nil
}

// NewClientWithOptions creates a new Docker client with custom options
func NewClientWithOptions(logger interfaces.Logger, opts ...client.Opt) (interfaces.DockerClient, error) {
	defaultOpts := []client.Opt{
		client.FromEnv,
		client.WithAPIVersionNegotiation(),
	}

	allOpts := append(defaultOpts, opts...)

	cli, err := client.NewClientWithOpts(allOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create docker client: %w", err)
	}

	return &Client{
		client: cli,
		logger: logger,
	}, nil
}

// IsRunning checks if Docker daemon is running
func (c *Client) IsRunning(ctx context.Context) bool {
	_, err := c.client.Ping(ctx)
	return err == nil
}

// ListContainers lists all containers
func (c *Client) ListContainers(ctx context.Context, all bool) ([]interfaces.ContainerInfo, error) {
	containers, err := c.client.ContainerList(ctx, container.ListOptions{
		All: all,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list containers: %w", err)
	}

	var result []interfaces.ContainerInfo
	for _, cont := range containers {
		info := interfaces.ContainerInfo{
			ID:      cont.ID[:12],
			Image:   cont.Image,
			Status:  cont.Status,
			State:   cont.State,
			Created: time.Unix(cont.Created, 0),
			Labels:  cont.Labels,
			Ports:   convertPorts(cont.Ports),
		}

		// Get container name
		if len(cont.Names) > 0 {
			info.Name = cont.Names[0]
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
		return fmt.Errorf("failed to read pull output: %w", err)
	}

	c.logger.Success("Image %s pulled successfully", imageName)
	return nil
}

// BuildImage builds a Docker image
func (c *Client) BuildImage(ctx context.Context, path string, tag string, buildArgs map[string]string) error {
	c.logger.Info("Building image %s from %s...", tag, path)

	buildOptions := types.ImageBuildOptions{
		Tags:      []string{tag},
		BuildArgs: convertBuildArgs(buildArgs),
		Remove:    true,
	}

	buildContext, err := createBuildContext(path)
	if err != nil {
		return fmt.Errorf("failed to create build context: %w", err)
	}

	resp, err := c.client.ImageBuild(ctx, buildContext, buildOptions)
	if err != nil {
		return fmt.Errorf("failed to build image: %w", err)
	}
	defer resp.Body.Close()

	// Read the output to ensure the build completes
	_, err = io.Copy(io.Discard, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read build output: %w", err)
	}

	c.logger.Success("Image %s built successfully", tag)
	return nil
}

// RunContainer runs a Docker container
func (c *Client) RunContainer(ctx context.Context, config interfaces.ContainerConfig) (string, error) {
	c.logger.Info("Running container %s from image %s...", config.Name, config.Image)

	// Create container
	containerConfig := &container.Config{
		Image:      config.Image,
		Cmd:        config.Cmd,
		Env:        config.Env,
		WorkingDir: config.WorkingDir,
		Labels:     config.Labels,
	}

	hostConfig := &container.HostConfig{
		AutoRemove:   config.AutoRemove,
		NetworkMode:  container.NetworkMode(config.NetworkMode),
		PortBindings: convertPortBindings(config.Ports),
		Binds:        convertVolumes(config.Volumes),
	}

	networkConfig := &network.NetworkingConfig{}

	resp, err := c.client.ContainerCreate(ctx, containerConfig, hostConfig, networkConfig, nil, config.Name)
	if err != nil {
		return "", fmt.Errorf("failed to create container: %w", err)
	}

	// Start container
	if err := c.client.ContainerStart(ctx, resp.ID, container.StartOptions{}); err != nil {
		return "", fmt.Errorf("failed to start container: %w", err)
	}

	c.logger.Success("Container %s started with ID %s", config.Name, resp.ID[:12])
	return resp.ID, nil
}

// StopContainer stops a running container
func (c *Client) StopContainer(ctx context.Context, containerID string, timeout time.Duration) error {
	c.logger.Info("Stopping container %s...", containerID[:12])

	timeoutSeconds := int(timeout.Seconds())
	err := c.client.ContainerStop(ctx, containerID, container.StopOptions{
		Timeout: &timeoutSeconds,
	})
	if err != nil {
		return fmt.Errorf("failed to stop container: %w", err)
	}

	c.logger.Success("Container %s stopped", containerID[:12])
	return nil
}

// RemoveContainer removes a container
func (c *Client) RemoveContainer(ctx context.Context, containerID string, force bool) error {
	c.logger.Info("Removing container %s...", containerID[:12])

	err := c.client.ContainerRemove(ctx, containerID, container.RemoveOptions{
		Force: force,
	})
	if err != nil {
		return fmt.Errorf("failed to remove container: %w", err)
	}

	c.logger.Success("Container %s removed", containerID[:12])
	return nil
}

// GetContainerLogs gets container logs
func (c *Client) GetContainerLogs(ctx context.Context, containerID string, follow bool) (io.ReadCloser, error) {
	options := container.LogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Follow:     follow,
		Timestamps: true,
	}

	reader, err := c.client.ContainerLogs(ctx, containerID, options)
	if err != nil {
		return nil, fmt.Errorf("failed to get container logs: %w", err)
	}

	return reader, nil
}

// WaitForContainer waits for a container to exit
func (c *Client) WaitForContainer(ctx context.Context, containerID string) (int, error) {
	c.logger.Info("Waiting for container %s to exit...", containerID[:12])

	statusCh, errCh := c.client.ContainerWait(ctx, containerID, container.WaitConditionNotRunning)

	select {
	case err := <-errCh:
		if err != nil {
			return -1, fmt.Errorf("error waiting for container: %w", err)
		}
	case status := <-statusCh:
		return int(status.StatusCode), nil
	}

	return -1, fmt.Errorf("unexpected wait result")
}

// InspectContainer inspects a container
func (c *Client) InspectContainer(ctx context.Context, containerID string) (*interfaces.ContainerInspect, error) {
	resp, err := c.client.ContainerInspect(ctx, containerID)
	if err != nil {
		return nil, fmt.Errorf("failed to inspect container: %w", err)
	}

	return convertInspectResponse(resp), nil
}

// Helper functions

func convertPorts(ports []types.Port) []interfaces.PortMapping {
	var result []interfaces.PortMapping
	for _, p := range ports {
		result = append(result, interfaces.PortMapping{
			Host:      fmt.Sprintf("%d", p.PublicPort),
			Container: fmt.Sprintf("%d", p.PrivatePort),
			Protocol:  p.Type,
		})
	}
	return result
}

func convertBuildArgs(args map[string]string) map[string]*string {
	result := make(map[string]*string)
	for k, v := range args {
		v := v // capture loop variable
		result[k] = &v
	}
	return result
}

func convertPortBindings(ports []interfaces.PortMapping) nat.PortMap {
	// Implementation depends on docker/go-connections/nat package
	// This is a simplified version
	return nil
}

func convertVolumes(volumes []interfaces.VolumeMount) []string {
	var result []string
	for _, v := range volumes {
		bind := fmt.Sprintf("%s:%s", v.Source, v.Target)
		if v.ReadOnly {
			bind += ":ro"
		}
		result = append(result, bind)
	}
	return result
}

func createBuildContext(path string) (io.Reader, error) {
	// Implementation would create a tar archive of the build context
	// This is a simplified version
	return nil, fmt.Errorf("not implemented")
}

func convertInspectResponse(resp types.ContainerJSON) *interfaces.ContainerInspect {
	startedAt, _ := time.Parse(time.RFC3339Nano, resp.State.StartedAt)
	finishedAt, _ := time.Parse(time.RFC3339Nano, resp.State.FinishedAt)

	return &interfaces.ContainerInspect{
		ID: resp.ID,
		State: interfaces.ContainerState{
			Status:     resp.State.Status,
			Running:    resp.State.Running,
			Paused:     resp.State.Paused,
			Restarting: resp.State.Restarting,
			OOMKilled:  resp.State.OOMKilled,
			Dead:       resp.State.Dead,
			Pid:        resp.State.Pid,
			ExitCode:   resp.State.ExitCode,
			Error:      resp.State.Error,
			StartedAt:  startedAt,
			FinishedAt: finishedAt,
		},
		// Additional fields would be converted here
	}
}
