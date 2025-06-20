package build

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"time"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/interfaces"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// Builder implements the interfaces.Builder interface
type Builder struct {
	provider BuildProvider
	cache    BuildCache
	logger   *slog.Logger
	tracer   trace.Tracer
	config   BuilderConfig
}

// BuilderConfig holds builder configuration
type BuilderConfig struct {
	EnableCache      bool
	CacheTTL         time.Duration
	DefaultPlatform  string
	DefaultBuildArgs map[string]string
	MaxConcurrency   int
}

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
	ID       string
	Tags     []string
	Size     int64
	Created  time.Time
	Platform string
	Digest   string
}

// BuildCache caches build artifacts
type BuildCache interface {
	Get(ctx context.Context, key string) (*CacheEntry, error)
	Set(ctx context.Context, key string, entry *CacheEntry) error
	Delete(ctx context.Context, key string) error
	Clear(ctx context.Context) error
}

// CacheEntry represents a cached build
type CacheEntry struct {
	ImageID   string
	Tags      []string
	BuildTime time.Time
	ExpiresAt time.Time
	Digest    string
	Size      int64
}

// NewBuilder creates a new builder
func NewBuilder(provider BuildProvider, cache BuildCache, logger *slog.Logger, config BuilderConfig) *Builder {
	return &Builder{
		provider: provider,
		cache:    cache,
		logger:   logger.With("component", "builder"),
		tracer:   otel.Tracer("builder"),
		config:   config,
	}
}

// Build implements interfaces.Builder
func (b *Builder) Build(ctx context.Context, options interfaces.BuildOptions) error {
	ctx, span := b.tracer.Start(ctx, "builder.build",
		trace.WithAttributes(
			attribute.String("image.name", options.ImageName),
			attribute.String("dockerfile", options.DockerfilePath),
			attribute.Bool("no_cache", options.NoCache),
		),
	)
	defer span.End()

	b.logger.InfoContext(ctx, "Building image",
		"image", options.ImageName,
		"dockerfile", options.DockerfilePath,
		"context", options.BuildContext,
	)

	// Check cache if enabled
	if b.config.EnableCache && !options.NoCache {
		cacheKey := b.generateCacheKey(options)
		if entry, err := b.cache.Get(ctx, cacheKey); err == nil && entry != nil {
			if time.Now().Before(entry.ExpiresAt) {
				b.logger.InfoContext(ctx, "Using cached image",
					"image", options.ImageName,
					"cached_id", entry.ImageID,
				)

				// Tag the cached image
				for _, tag := range entry.Tags {
					if tag == options.ImageName {
						return nil // Already tagged
					}
				}

				// Tag with new name
				if len(entry.Tags) > 0 {
					return b.provider.Tag(ctx, entry.Tags[0], options.ImageName)
				}
			}
		}
	}

	// Merge build args
	buildArgs := make(map[string]string)
	for k, v := range b.config.DefaultBuildArgs {
		buildArgs[k] = v
	}
	for k, v := range options.BuildArgs {
		buildArgs[k] = v
	}

	// Set platform
	platform := options.Platform
	if platform == "" {
		platform = b.config.DefaultPlatform
	}

	// Build the image
	providerOpts := ProviderBuildOptions{
		Dockerfile: options.DockerfilePath,
		Context:    options.BuildContext,
		Tags:       []string{options.ImageName},
		BuildArgs:  buildArgs,
		Target:     options.Target,
		NoCache:    options.NoCache,
		Platform:   platform,
	}

	startTime := time.Now()
	err := b.provider.Build(ctx, providerOpts)
	buildDuration := time.Since(startTime)

	if err != nil {
		span.RecordError(err)
		b.logger.ErrorContext(ctx, "Build failed",
			"image", options.ImageName,
			"error", err,
			"duration", buildDuration,
		)
		return fmt.Errorf("build failed: %w", err)
	}

	b.logger.InfoContext(ctx, "Build completed",
		"image", options.ImageName,
		"duration", buildDuration,
	)

	// Cache the build if enabled
	if b.config.EnableCache {
		b.cacheImage(ctx, options)
	}

	return nil
}

// Push implements interfaces.Builder
func (b *Builder) Push(ctx context.Context, image string) error {
	ctx, span := b.tracer.Start(ctx, "builder.push",
		trace.WithAttributes(
			attribute.String("image", image),
		),
	)
	defer span.End()

	b.logger.InfoContext(ctx, "Pushing image", "image", image)

	// Extract registry from image name
	auth := b.getAuthForImage(image)

	startTime := time.Now()
	err := b.provider.Push(ctx, image, auth)
	pushDuration := time.Since(startTime)

	if err != nil {
		span.RecordError(err)
		b.logger.ErrorContext(ctx, "Push failed",
			"image", image,
			"error", err,
			"duration", pushDuration,
		)
		return fmt.Errorf("push failed: %w", err)
	}

	b.logger.InfoContext(ctx, "Push completed",
		"image", image,
		"duration", pushDuration,
	)

	return nil
}

// Tag implements interfaces.Builder
func (b *Builder) Tag(ctx context.Context, source, target string) error {
	ctx, span := b.tracer.Start(ctx, "builder.tag",
		trace.WithAttributes(
			attribute.String("source", source),
			attribute.String("target", target),
		),
	)
	defer span.End()

	b.logger.InfoContext(ctx, "Tagging image",
		"source", source,
		"target", target,
	)

	err := b.provider.Tag(ctx, source, target)
	if err != nil {
		span.RecordError(err)
		return fmt.Errorf("tag failed: %w", err)
	}

	return nil
}

// Pull implements interfaces.Builder
func (b *Builder) Pull(ctx context.Context, image string) error {
	ctx, span := b.tracer.Start(ctx, "builder.pull",
		trace.WithAttributes(
			attribute.String("image", image),
		),
	)
	defer span.End()

	b.logger.InfoContext(ctx, "Pulling image", "image", image)

	// Check if image already exists
	exists, err := b.provider.ImageExists(ctx, image)
	if err != nil {
		b.logger.WarnContext(ctx, "Failed to check image existence",
			"image", image,
			"error", err,
		)
	} else if exists {
		b.logger.InfoContext(ctx, "Image already exists locally", "image", image)
		return nil
	}

	// Extract registry from image name
	auth := b.getAuthForImage(image)

	startTime := time.Now()
	err = b.provider.Pull(ctx, image, auth)
	pullDuration := time.Since(startTime)

	if err != nil {
		span.RecordError(err)
		b.logger.ErrorContext(ctx, "Pull failed",
			"image", image,
			"error", err,
			"duration", pullDuration,
		)
		return fmt.Errorf("pull failed: %w", err)
	}

	b.logger.InfoContext(ctx, "Pull completed",
		"image", image,
		"duration", pullDuration,
	)

	return nil
}

// Exists implements interfaces.Builder
func (b *Builder) Exists(ctx context.Context, image string) (bool, error) {
	ctx, span := b.tracer.Start(ctx, "builder.exists",
		trace.WithAttributes(
			attribute.String("image", image),
		),
	)
	defer span.End()

	exists, err := b.provider.ImageExists(ctx, image)
	if err != nil {
		span.RecordError(err)
		return false, fmt.Errorf("failed to check image existence: %w", err)
	}

	return exists, nil
}

// Remove implements interfaces.Builder
func (b *Builder) Remove(ctx context.Context, image string) error {
	ctx, span := b.tracer.Start(ctx, "builder.remove",
		trace.WithAttributes(
			attribute.String("image", image),
		),
	)
	defer span.End()

	b.logger.InfoContext(ctx, "Removing image", "image", image)

	err := b.provider.RemoveImage(ctx, image)
	if err != nil {
		span.RecordError(err)
		return fmt.Errorf("remove failed: %w", err)
	}

	// Remove from cache if present
	if b.config.EnableCache {
		cacheKey := b.generateCacheKeyForImage(image)
		if err := b.cache.Delete(ctx, cacheKey); err != nil {
			b.logger.WarnContext(ctx, "Failed to remove from cache",
				"image", image,
				"error", err,
			)
		}
	}

	return nil
}

// List implements interfaces.Builder
func (b *Builder) List(ctx context.Context, pattern string) ([]string, error) {
	ctx, span := b.tracer.Start(ctx, "builder.list",
		trace.WithAttributes(
			attribute.String("pattern", pattern),
		),
	)
	defer span.End()

	images, err := b.provider.ListImages(ctx)
	if err != nil {
		span.RecordError(err)
		return nil, fmt.Errorf("list failed: %w", err)
	}

	var result []string
	for _, img := range images {
		for _, tag := range img.Tags {
			if pattern == "" || matchesPattern(tag, pattern) {
				result = append(result, tag)
			}
		}
	}

	return result, nil
}

// generateCacheKey generates a cache key for build options
func (b *Builder) generateCacheKey(options interfaces.BuildOptions) string {
	parts := []string{
		"build",
		options.DockerfilePath,
		options.BuildContext,
		options.Target,
		options.Platform,
	}

	// Add sorted build args
	for k, v := range options.BuildArgs {
		parts = append(parts, fmt.Sprintf("%s=%s", k, v))
	}

	return strings.Join(parts, ":")
}

// generateCacheKeyForImage generates a cache key for an image
func (b *Builder) generateCacheKeyForImage(image string) string {
	return fmt.Sprintf("image:%s", image)
}

// cacheImage caches the built image
func (b *Builder) cacheImage(ctx context.Context, options interfaces.BuildOptions) {
	images, err := b.provider.ListImages(ctx)
	if err != nil {
		b.logger.WarnContext(ctx, "Failed to list images for caching",
			"error", err,
		)
		return
	}

	// Find the built image
	for _, img := range images {
		for _, tag := range img.Tags {
			if tag == options.ImageName {
				entry := &CacheEntry{
					ImageID:   img.ID,
					Tags:      img.Tags,
					BuildTime: time.Now(),
					ExpiresAt: time.Now().Add(b.config.CacheTTL),
					Digest:    img.Digest,
					Size:      img.Size,
				}

				cacheKey := b.generateCacheKey(options)
				if err := b.cache.Set(ctx, cacheKey, entry); err != nil {
					b.logger.WarnContext(ctx, "Failed to cache image",
						"image", options.ImageName,
						"error", err,
					)
				}
				return
			}
		}
	}
}

// getAuthForImage extracts auth config for an image
func (b *Builder) getAuthForImage(image string) *AuthConfig {
	// In a real implementation, this would look up credentials
	// from Docker config or a secret manager
	return nil
}

// matchesPattern checks if a string matches a pattern
func matchesPattern(s, pattern string) bool {
	// Simple pattern matching - in real implementation use glob
	if pattern == "*" {
		return true
	}

	// Check prefix
	if strings.HasSuffix(pattern, "*") {
		prefix := strings.TrimSuffix(pattern, "*")
		return strings.HasPrefix(s, prefix)
	}

	// Check suffix
	if strings.HasPrefix(pattern, "*") {
		suffix := strings.TrimPrefix(pattern, "*")
		return strings.HasSuffix(s, suffix)
	}

	// Exact match
	return s == pattern
}
