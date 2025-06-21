package docker

import (
	"context"
	"io"
)

// BuildProvider defines the underlying build provider interface
type BuildProvider interface {
	Build(ctx context.Context, options ProviderBuildOptions) error
	Push(ctx context.Context, image string, auth *AuthConfig) error
	Tag(ctx context.Context, source, target string) error
	Pull(ctx context.Context, image string, auth *AuthConfig) error
	ImageExists(ctx context.Context, image string) (bool, error)
	RemoveImage(ctx context.Context, image string) error
	ListImages(ctx context.Context) ([]ImageInfo, error)
}

// ProviderBuildOptions contains provider-specific build options
type ProviderBuildOptions struct {
	Dockerfile   string
	Context      string
	Tags         []string
	BuildArgs    map[string]string
	Target       string
	NoCache      bool
	Platform     string
	OutputStream io.Writer
}

// AuthConfig contains registry authentication
type AuthConfig struct {
	Username string
	Password string
	Server   string
}

// ImageInfo contains image information
type ImageInfo struct {
	ID      string
	Tags    []string
	Size    int64
	Created int64
}
