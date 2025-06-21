package docker

import (
	"context"
	"log/slog"
	"os/exec"
	"testing"
	"time"
)

// isDockerAvailable checks if Docker daemon is accessible
func isDockerAvailable() bool {
	cmd := exec.Command("docker", "version")
	err := cmd.Run()
	return err == nil
}

func TestNewCLIProvider(t *testing.T) {
	logger := slog.Default()
	provider := NewCLIProvider(logger)

	if provider == nil {
		t.Fatal("NewProvider returned nil")
	}

	if provider.binaryPath != "docker" {
		t.Errorf("NewCLIProvider() binaryPath = %v, want %v", provider.binaryPath, "docker")
	}

	if provider.logger == nil {
		t.Error("NewCLIProvider() logger is nil")
	}
}

func TestProvider_Build(t *testing.T) {
	logger := slog.Default()
	provider := NewCLIProvider(logger)
	ctx := context.Background()

	if isDockerAvailable() {
		// Test with Docker available
		t.Run("with_docker", func(t *testing.T) {
			// Build will fail because Dockerfile doesn't exist in test directory
			options := ProviderBuildOptions{
				Dockerfile: "Dockerfile",
				Context:    ".",
				Tags:       []string{"test:latest"},
			}
			err := provider.Build(ctx, options)
			if err == nil {
				t.Error("Build() expected error for missing Dockerfile")
			}
		})

		// Test with build args
		t.Run("with_docker_build_args", func(t *testing.T) {
			options := ProviderBuildOptions{
				Dockerfile: "Dockerfile",
				Context:    ".",
				Tags:       []string{"test:latest", "test:v1.0"},
				BuildArgs: map[string]string{
					"VERSION": "1.0",
					"ENV":     "prod",
				},
				NoCache: true,
			}
			err := provider.Build(ctx, options)
			if err == nil {
				t.Error("Build() expected error for missing Dockerfile")
			}
		})
	} else {
		// Test without Docker
		t.Run("without_docker", func(t *testing.T) {
			options := ProviderBuildOptions{
				Dockerfile: "Dockerfile",
				Context:    ".",
				Tags:       []string{"test:latest"},
			}
			err := provider.Build(ctx, options)
			if err == nil {
				t.Error("Build() expected error without docker daemon")
			}
		})
	}
}

func TestProvider_Push(t *testing.T) {
	logger := slog.Default()
	provider := NewCLIProvider(logger)
	ctx := context.Background()

	if isDockerAvailable() {
		// Test with Docker available
		t.Run("with_docker", func(t *testing.T) {
			// Push will fail because image doesn't exist
			err := provider.Push(ctx, "nonexistent:image:tag:123456", nil)
			if err == nil {
				t.Error("Push() expected error for non-existent image")
			}
		})

		// Test with auth
		t.Run("with_docker_auth", func(t *testing.T) {
			auth := &AuthConfig{
				Username: "testuser",
				Password: "testpass",
				Server:   "localhost:5000",
			}
			// This will fail due to invalid credentials
			err := provider.Push(ctx, "test:latest", auth)
			if err == nil {
				t.Error("Push() expected error with invalid auth")
			}
		})
	} else {
		// Test without Docker
		t.Run("without_docker", func(t *testing.T) {
			err := provider.Push(ctx, "test:latest", nil)
			if err == nil {
				t.Error("Push() expected error without docker daemon")
			}
		})
	}
}

func TestProvider_Tag(t *testing.T) {
	logger := slog.Default()
	provider := NewCLIProvider(logger)
	ctx := context.Background()

	if isDockerAvailable() {
		// Test with Docker available
		t.Run("with_docker", func(t *testing.T) {
			// Tag will fail because source image doesn't exist
			err := provider.Tag(ctx, "nonexistent:source:123456", "test:v1.0")
			if err == nil {
				t.Error("Tag() expected error for non-existent source image")
			}
		})
	} else {
		// Test without Docker
		t.Run("without_docker", func(t *testing.T) {
			err := provider.Tag(ctx, "test:latest", "test:v1.0")
			if err == nil {
				t.Error("Tag() expected error without docker daemon")
			}
		})
	}
}

func TestProvider_Pull(t *testing.T) {
	logger := slog.Default()
	provider := NewCLIProvider(logger)
	ctx := context.Background()

	if isDockerAvailable() {
		// Test with Docker available
		t.Run("with_docker", func(t *testing.T) {
			// Use a non-existent image to avoid pulling
			err := provider.Pull(ctx, "nonexistent:image:that:does:not:exist:123456", nil)
			if err == nil {
				t.Error("Pull() expected error for non-existent image")
			}
		})

		// Test with auth
		t.Run("with_docker_auth", func(t *testing.T) {
			auth := &AuthConfig{
				Username: "testuser",
				Password: "testpass",
				Server:   "localhost:5000",
			}
			// This will fail due to invalid credentials
			err := provider.Pull(ctx, "localhost:5000/test:latest", auth)
			if err == nil {
				t.Error("Pull() expected error with invalid auth")
			}
		})
	} else {
		// Test without Docker
		t.Run("without_docker", func(t *testing.T) {
			err := provider.Pull(ctx, "alpine:latest", nil)
			if err == nil {
				t.Error("Pull() expected error without docker daemon")
			}
		})
	}
}

func TestProvider_ImageExists(t *testing.T) {
	logger := slog.Default()
	provider := NewCLIProvider(logger)
	ctx := context.Background()

	if isDockerAvailable() {
		// Test with Docker available
		t.Run("with_docker", func(t *testing.T) {
			// Check for a non-existent image
			exists, err := provider.ImageExists(ctx, "nonexistent:image:tag:123456")
			if err != nil {
				t.Errorf("ImageExists() unexpected error: %v", err)
			}
			if exists {
				t.Error("ImageExists() should return false for non-existent image")
			}
		})
	} else {
		// Test without Docker
		t.Run("without_docker", func(t *testing.T) {
			exists, err := provider.ImageExists(ctx, "test:latest")
			// Note: ImageExists returns (false, nil) when docker command fails
			// This is by design - it treats "image not found" as non-error case
			if err != nil {
				t.Errorf("ImageExists() unexpected error: %v", err)
			}
			if exists {
				t.Error("ImageExists() should return false without docker daemon")
			}
		})
	}
}

func TestProvider_RemoveImage(t *testing.T) {
	logger := slog.Default()
	provider := NewCLIProvider(logger)
	ctx := context.Background()

	if isDockerAvailable() {
		// Test with Docker available
		t.Run("with_docker", func(t *testing.T) {
			// RemoveImage will fail because image doesn't exist
			err := provider.RemoveImage(ctx, "nonexistent:image:tag:123456")
			if err == nil {
				t.Error("RemoveImage() expected error for non-existent image")
			}
		})
	} else {
		// Test without Docker
		t.Run("without_docker", func(t *testing.T) {
			err := provider.RemoveImage(ctx, "test:latest")
			if err == nil {
				t.Error("RemoveImage() expected error without docker daemon")
			}
		})
	}
}

func TestProvider_ListImages(t *testing.T) {
	logger := slog.Default()
	provider := NewCLIProvider(logger)
	ctx := context.Background()

	if isDockerAvailable() {
		// Test with Docker available
		t.Run("with_docker", func(t *testing.T) {
			images, err := provider.ListImages(ctx)
			if err != nil {
				t.Errorf("ListImages() unexpected error: %v", err)
			}
			// Should return at least an empty list
			if images == nil {
				t.Error("ListImages() should not return nil slice")
			}
		})
	} else {
		// Test without Docker
		t.Run("without_docker", func(t *testing.T) {
			images, err := provider.ListImages(ctx)
			if err == nil {
				t.Error("ListImages() expected error without docker daemon")
			}
			if len(images) != 0 {
				t.Error("ListImages() should return empty list on error")
			}
		})
	}
}

func TestImageInfo(t *testing.T) {
	// Test ImageInfo struct
	info := ImageInfo{
		ID:      "sha256:abc123",
		Tags:    []string{"test:latest", "test:v1.0"},
		Size:    1024 * 1024 * 100, // 100MB
		Created: time.Now().Unix(),
	}

	if info.ID != "sha256:abc123" {
		t.Errorf("ImageInfo.ID = %v", info.ID)
	}

	if len(info.Tags) != 2 {
		t.Errorf("ImageInfo.Tags length = %d, want 2", len(info.Tags))
	}

	if info.Size != 1024*1024*100 {
		t.Errorf("ImageInfo.Size = %v", info.Size)
	}
}

func TestAuthConfig(t *testing.T) {
	// Test AuthConfig struct
	auth := &AuthConfig{
		Username: "user",
		Password: "pass",
		Server:   "docker.io",
	}

	if auth.Username != "user" {
		t.Errorf("AuthConfig.Username = %v", auth.Username)
	}

	if auth.Password != "pass" {
		t.Errorf("AuthConfig.Password = %v", auth.Password)
	}

	if auth.Server != "docker.io" {
		t.Errorf("AuthConfig.Server = %v", auth.Server)
	}
}

func TestProvider_parseSize(t *testing.T) {
	tests := []struct {
		name     string
		sizeStr  string
		wantSize int64
	}{
		{"bytes", "1024", 1024},
		{"kilobytes", "2KB", 2048},
		{"megabytes", "3MB", 3145728},
		{"gigabytes", "1GB", 1073741824},
		{"lowercase_kb", "5kb", 5120},
		{"lowercase_mb", "10mb", 10485760},
		{"lowercase_gb", "2gb", 2147483648},
		{"decimal", "1.5GB", 1610612736},
		{"invalid", "invalid", 0},
		{"empty", "", 0},
		{"with_spaces", " 100 MB ", 104857600},
	}

	provider := &CLIProvider{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := provider.parseSize(tt.sizeStr)
			if got != tt.wantSize {
				t.Errorf("parseSize(%q) = %d, want %d", tt.sizeStr, got, tt.wantSize)
			}
		})
	}
}

func TestProvider_login(t *testing.T) {
	logger := slog.Default()
	provider := NewCLIProvider(logger)
	ctx := context.Background()

	tests := []struct {
		name    string
		auth    *AuthConfig
		wantErr bool
	}{
		{
			name: "full_auth",
			auth: &AuthConfig{
				Username: "user",
				Password: "pass",
				Server:   "docker.io",
			},
			wantErr: true, // Will fail with invalid credentials
		},
		{
			name: "username_only",
			auth: &AuthConfig{
				Username: "user",
			},
			wantErr: true,
		},
		{
			name:    "empty_auth",
			auth:    &AuthConfig{},
			wantErr: true, // Docker login with empty auth will fail
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !isDockerAvailable() {
				t.Skip("Docker not available")
			}

			err := provider.login(ctx, tt.auth)
			if (err != nil) != tt.wantErr {
				t.Errorf("login() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestProvider_logout(t *testing.T) {
	logger := slog.Default()
	provider := NewCLIProvider(logger)
	ctx := context.Background()

	tests := []struct {
		name   string
		server string
	}{
		{"with_server", "docker.io"},
		{"empty_server", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !isDockerAvailable() {
				t.Skip("Docker not available")
			}

			// logout always returns nil
			err := provider.logout(ctx, tt.server)
			if err != nil {
				t.Errorf("logout() unexpected error = %v", err)
			}
		})
	}
}

// Benchmark tests
func BenchmarkNewCLIProvider(b *testing.B) {
	logger := slog.Default()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = NewCLIProvider(logger)
	}
}

func BenchmarkProvider_Build(b *testing.B) {
	logger := slog.Default()
	provider := NewCLIProvider(logger)
	ctx := context.Background()
	options := ProviderBuildOptions{
		Dockerfile: "Dockerfile",
		Context:    ".",
		Tags:       []string{"bench:latest"},
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = provider.Build(ctx, options)
	}
}
