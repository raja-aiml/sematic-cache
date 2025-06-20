package interfaces

import (
	"context"
)

// Builder defines the interface for building container images
type Builder interface {
	// Build builds a container image
	Build(ctx context.Context, options BuildOptions) error
	
	// Push pushes an image to a registry
	Push(ctx context.Context, image string) error
	
	// Tag tags an image
	Tag(ctx context.Context, source, target string) error
	
	// Pull pulls an image from a registry
	Pull(ctx context.Context, image string) error
	
	// Exists checks if an image exists locally
	Exists(ctx context.Context, image string) (bool, error)
	
	// Remove removes an image
	Remove(ctx context.Context, image string) error
	
	// List lists images matching a pattern
	List(ctx context.Context, pattern string) ([]string, error)
}

// BuildOptions contains options for building an image
type BuildOptions struct {
	// ImageName is the name of the image to build
	ImageName string
	
	// DockerfilePath is the path to the Dockerfile
	DockerfilePath string
	
	// BuildContext is the build context directory
	BuildContext string
	
	// BuildArgs are build arguments
	BuildArgs map[string]string
	
	// Target is the build target
	Target string
	
	// NoCache disables build cache
	NoCache bool
	
	// Platform is the target platform
	Platform string
}