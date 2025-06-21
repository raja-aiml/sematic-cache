# Docker Package

This package provides a unified interface for Docker operations using the Docker SDK.

## Overview

The `docker` package provides a single, consistent implementation using only the Docker SDK for better maintainability and performance.

## Builder

The `Builder` is the main type that implements all Docker operations using the Docker SDK:

```go
// Create a new builder
builder, err := docker.NewBuilder()
if err != nil {
    return err
}
defer builder.Close()

// Build an image
err = builder.Build(ctx, docker.ProviderBuildOptions{
    Dockerfile: "Dockerfile",
    Context:    ".",
    Tags:       []string{"myapp:latest"},
})

// Or use the simple interface
err = builder.BuildSimple(ctx, "Dockerfile", "myapp:latest", ".")
```

## Features

- **SDK-Only Implementation**: All operations use the official Docker Go SDK
- **BuildProvider Interface**: Implements the standard `BuildProvider` interface
- **Comprehensive Operations**: Build, push, pull, tag, run, stop, remove containers
- **K3d Integration**: Import images to k3d clusters (uses CLI as k3d has no SDK)
- **Streaming Output**: Real-time build and operation output
- **Proper Resource Management**: Includes `Close()` method for cleanup

## Migration Guide

## API Reference

The `Builder` type provides the following methods:

- `Build()` - Build Docker images with full options
- `BuildSimple()` - Simplified build interface
- `Push()` - Push images to registry
- `Pull()` - Pull images from registry
- `Tag()` - Tag images
- `ImageExists()` - Check if image exists
- `RemoveImage()` - Remove images
- `ListImages()` - List all images
- `ImportToK3d()` - Import images to k3d clusters
- `RunContainer()` - Run containers
- `StopContainer()` - Stop running containers
- `RemoveContainer()` - Remove containers
- `GetContainerLogs()` - Get container logs
- `Close()` - Clean up resources

## Container Operations

```go
// Run a container
config := &docker.ContainerConfig{
    Name:       "mycontainer",
    Cmd:        []string{"echo", "hello"},
    Env:        []string{"FOO=bar"},
    Volumes:    []string{"/host/path:/container/path"},
    Ports:      map[string]string{"8080": "80"},
    AutoRemove: true,
}

containerID, err := builder.RunContainer(ctx, "myimage", config)

// Get logs
logs, err := builder.GetContainerLogs(ctx, containerID, true) // follow=true

// Stop container
err = builder.StopContainer(ctx, containerID)

// Remove container
err = builder.RemoveContainer(ctx, containerID)
```

## Best Practices

1. Always call `Close()` when done with the builder
2. Use context for cancellation
3. Handle streaming output appropriately
4. Check `ImageExists()` before operations that require an image
5. Use structured `ContainerConfig` for container operations

## Testing

The package includes comprehensive tests. Docker daemon must be available for tests to run:

```bash
go test ./pkg/docker
```

To skip Docker-dependent tests when Docker is not available:

```go
if !isDockerAvailable() {
    t.Skip("Docker daemon not available")
}