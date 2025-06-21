package docker

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/api/types/registry"
	"github.com/docker/docker/client"
	"github.com/docker/go-connections/nat"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
)

// Builder implements all Docker operations using only the Docker SDK
type Builder struct {
	client *client.Client
	logger *utils.Logger
}

// NewBuilder creates a new Docker builder using only SDK
func NewBuilder() (*Builder, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, fmt.Errorf("failed to create Docker client: %w", err)
	}

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if _, err := cli.Ping(ctx); err != nil {
		cli.Close()
		return nil, fmt.Errorf("failed to connect to Docker daemon: %w", err)
	}

	return &Builder{
		client: cli,
		logger: utils.NewLogger("docker"),
	}, nil
}

// Build implements BuildProvider interface using SDK
func (b *Builder) Build(ctx context.Context, options ProviderBuildOptions) error {
	b.logger.Info("Building image with Docker SDK, tags: %v", options.Tags)

	// Create build context
	buildCtx, err := b.createBuildContext(options.Context, options.Dockerfile)
	if err != nil {
		return fmt.Errorf("failed to create build context: %w", err)
	}

	// Prepare build options
	// Convert BuildArgs from map[string]string to map[string]*string
	buildArgs := make(map[string]*string)
	for k, v := range options.BuildArgs {
		val := v
		buildArgs[k] = &val
	}

	buildOpts := types.ImageBuildOptions{
		Tags:        options.Tags,
		Dockerfile:  filepath.Base(options.Dockerfile),
		BuildArgs:   buildArgs,
		NoCache:     options.NoCache,
		Target:      options.Target,
		Platform:    options.Platform,
		Remove:      true,
		ForceRemove: true,
	}

	// Build the image
	resp, err := b.client.ImageBuild(ctx, buildCtx, buildOpts)
	if err != nil {
		return fmt.Errorf("failed to build image: %w", err)
	}
	defer resp.Body.Close()

	// Stream build output
	return b.streamBuildOutput(resp.Body, options.OutputStream)
}

// BuildSimple provides a simple build interface for backward compatibility
func (b *Builder) BuildSimple(ctx context.Context, dockerfilePath, imageName, buildContext string) error {
	return b.Build(ctx, ProviderBuildOptions{
		Dockerfile:   dockerfilePath,
		Context:      buildContext,
		Tags:         []string{imageName},
		OutputStream: os.Stdout,
	})
}

// Push pushes an image to a registry using SDK
func (b *Builder) Push(ctx context.Context, imageName string, auth *AuthConfig) error {
	b.logger.Info("Pushing image: %s", imageName)

	var pushOpts image.PushOptions

	if auth != nil {
		authBytes, _ := json.Marshal(registry.AuthConfig{
			Username:      auth.Username,
			Password:      auth.Password,
			ServerAddress: auth.Server,
		})
		pushOpts.RegistryAuth = base64.URLEncoding.EncodeToString(authBytes)
	}

	resp, err := b.client.ImagePush(ctx, imageName, pushOpts)
	if err != nil {
		return fmt.Errorf("failed to push image: %w", err)
	}
	defer resp.Close()

	return b.streamOutput(resp, os.Stdout)
}

// Tag tags an image using SDK
func (b *Builder) Tag(ctx context.Context, source, target string) error {
	b.logger.Info("Tagging image %s as %s", source, target)

	if err := b.client.ImageTag(ctx, source, target); err != nil {
		return fmt.Errorf("failed to tag image: %w", err)
	}

	return nil
}

// Pull pulls an image using SDK
func (b *Builder) Pull(ctx context.Context, imageName string, auth *AuthConfig) error {
	b.logger.Info("Pulling image: %s", imageName)

	var pullOpts image.PullOptions

	if auth != nil {
		authBytes, _ := json.Marshal(registry.AuthConfig{
			Username:      auth.Username,
			Password:      auth.Password,
			ServerAddress: auth.Server,
		})
		pullOpts.RegistryAuth = base64.URLEncoding.EncodeToString(authBytes)
	}

	resp, err := b.client.ImagePull(ctx, imageName, pullOpts)
	if err != nil {
		return fmt.Errorf("failed to pull image: %w", err)
	}
	defer resp.Close()

	return b.streamOutput(resp, os.Stdout)
}

// ImageExists checks if an image exists using SDK
func (b *Builder) ImageExists(ctx context.Context, imageName string) (bool, error) {
	images, err := b.client.ImageList(ctx, image.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("failed to list images: %w", err)
	}

	for _, img := range images {
		for _, tag := range img.RepoTags {
			if tag == imageName {
				return true, nil
			}
		}
	}

	return false, nil
}

// RemoveImage removes an image using SDK
func (b *Builder) RemoveImage(ctx context.Context, imageName string) error {
	b.logger.Info("Removing image: %s", imageName)

	_, err := b.client.ImageRemove(ctx, imageName, image.RemoveOptions{
		Force:         true,
		PruneChildren: true,
	})
	if err != nil {
		return fmt.Errorf("failed to remove image: %w", err)
	}

	return nil
}

// ListImages lists all images using SDK
func (b *Builder) ListImages(ctx context.Context) ([]ImageInfo, error) {
	images, err := b.client.ImageList(ctx, image.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to list images: %w", err)
	}

	var result []ImageInfo
	for _, img := range images {
		result = append(result, ImageInfo{
			ID:      img.ID,
			Tags:    img.RepoTags,
			Size:    img.Size,
			Created: img.Created,
		})
	}

	return result, nil
}

// ImportToK3d imports an image to K3d cluster using k3d CLI
// Note: This is the only operation that requires CLI as k3d doesn't have an SDK
func (b *Builder) ImportToK3d(ctx context.Context, imageName, clusterName string) error {
	b.logger.Info("Importing image %s to k3d cluster %s", imageName, clusterName)

	// Check if image exists first
	exists, err := b.ImageExists(ctx, imageName)
	if err != nil {
		return fmt.Errorf("failed to check image existence: %w", err)
	}
	if !exists {
		return fmt.Errorf("image %s does not exist", imageName)
	}

	// k3d requires CLI - no SDK available
	cmd := []string{"k3d", "image", "import", imageName, "--cluster", clusterName}
	_, err = utils.RunCommand(ctx, cmd[0], cmd[1:], nil)
	return err
}

// RunContainer runs a container using SDK
func (b *Builder) RunContainer(ctx context.Context, imageName string, config *ContainerConfig) (string, error) {
	b.logger.Info("Running container from image: %s", imageName)

	// Container configuration
	containerConfig := &container.Config{
		Image:        imageName,
		Cmd:          config.Cmd,
		Env:          config.Env,
		WorkingDir:   config.WorkingDir,
		AttachStdout: true,
		AttachStderr: true,
	}

	// Host configuration
	hostConfig := &container.HostConfig{
		AutoRemove: config.AutoRemove,
	}

	// Port bindings
	if len(config.Ports) > 0 {
		containerConfig.ExposedPorts = nat.PortSet{}
		hostConfig.PortBindings = nat.PortMap{}

		for hostPort, containerPort := range config.Ports {
			port := nat.Port(fmt.Sprintf("%s/tcp", containerPort))
			containerConfig.ExposedPorts[port] = struct{}{}
			hostConfig.PortBindings[port] = []nat.PortBinding{
				{HostPort: hostPort},
			}
		}
	}

	// Volume mounts
	hostConfig.Binds = config.Volumes

	// Create container
	resp, err := b.client.ContainerCreate(ctx, containerConfig, hostConfig, nil, nil, config.Name)
	if err != nil {
		return "", fmt.Errorf("failed to create container: %w", err)
	}

	// Start container
	if err := b.client.ContainerStart(ctx, resp.ID, container.StartOptions{}); err != nil {
		return "", fmt.Errorf("failed to start container: %w", err)
	}

	return resp.ID, nil
}

// StopContainer stops a container using SDK
func (b *Builder) StopContainer(ctx context.Context, containerID string) error {
	timeout := 10
	stopOptions := container.StopOptions{
		Timeout: &timeout,
	}

	if err := b.client.ContainerStop(ctx, containerID, stopOptions); err != nil {
		return fmt.Errorf("failed to stop container: %w", err)
	}

	return nil
}

// RemoveContainer removes a container using SDK
func (b *Builder) RemoveContainer(ctx context.Context, containerID string) error {
	removeOptions := container.RemoveOptions{
		Force:         true,
		RemoveVolumes: true,
	}

	if err := b.client.ContainerRemove(ctx, containerID, removeOptions); err != nil {
		return fmt.Errorf("failed to remove container: %w", err)
	}

	return nil
}

// GetContainerLogs gets container logs using SDK
func (b *Builder) GetContainerLogs(ctx context.Context, containerID string, follow bool) (io.ReadCloser, error) {
	options := container.LogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Follow:     follow,
		Timestamps: true,
	}

	logs, err := b.client.ContainerLogs(ctx, containerID, options)
	if err != nil {
		return nil, fmt.Errorf("failed to get container logs: %w", err)
	}

	return logs, nil
}

// Close closes the Docker client connection
func (b *Builder) Close() error {
	if b.client != nil {
		return b.client.Close()
	}
	return nil
}

// Helper methods

func (b *Builder) createBuildContext(contextPath, dockerfilePath string) (io.Reader, error) {
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	defer tw.Close()

	// Add Dockerfile
	dockerfileContent, err := os.ReadFile(dockerfilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read Dockerfile: %w", err)
	}

	dockerfileHeader := &tar.Header{
		Name: "Dockerfile",
		Mode: 0644,
		Size: int64(len(dockerfileContent)),
	}

	if err := tw.WriteHeader(dockerfileHeader); err != nil {
		return nil, err
	}

	if _, err := tw.Write(dockerfileContent); err != nil {
		return nil, err
	}

	// Add context files
	err = filepath.Walk(contextPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip directories and the Dockerfile
		if info.IsDir() || path == dockerfilePath {
			return nil
		}

		// Create relative path
		relPath, err := filepath.Rel(contextPath, path)
		if err != nil {
			return err
		}

		// Read file
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}

		// Create tar header
		header := &tar.Header{
			Name: relPath,
			Mode: 0644,
			Size: int64(len(data)),
		}

		// Write header and data
		if err := tw.WriteHeader(header); err != nil {
			return err
		}

		if _, err := tw.Write(data); err != nil {
			return err
		}

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to create build context: %w", err)
	}

	return &buf, nil
}

func (b *Builder) streamBuildOutput(reader io.Reader, output io.Writer) error {
	decoder := json.NewDecoder(reader)

	for {
		var message struct {
			Stream string `json:"stream"`
			Error  string `json:"error"`
		}

		if err := decoder.Decode(&message); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		if message.Error != "" {
			return fmt.Errorf("build error: %s", message.Error)
		}

		if message.Stream != "" && output != nil {
			fmt.Fprint(output, message.Stream)
		}
	}

	return nil
}

func (b *Builder) streamOutput(reader io.Reader, output io.Writer) error {
	if output == nil {
		output = io.Discard
	}

	_, err := io.Copy(output, reader)
	return err
}

// ContainerConfig holds container runtime configuration
type ContainerConfig struct {
	Name       string
	Cmd        []string
	Env        []string
	WorkingDir string
	Volumes    []string          // host:container format
	Ports      map[string]string // host:container format
	AutoRemove bool
}
