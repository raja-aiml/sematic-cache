package docker

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
)

func TestNewBuilder(t *testing.T) {
	builder := NewBuilder()

	if builder == nil {
		t.Fatal("NewBuilder() returned nil")
	}

	if builder.logger == nil {
		t.Error("Builder logger is nil")
	}

	// SDK initialization is attempted by default
	if !builder.useSDK {
		// This is OK - SDK might not be available in test environment
		t.Log("SDK not available, using CLI fallback")
	}
}

func TestBuilder_IsDockerRunning(t *testing.T) {
	builder := &Builder{
		logger: utils.NewLogger("test"),
		useSDK: false, // Use CLI to avoid SDK dependency
	}

	ctx := context.Background()
	// This will check if docker CLI is available
	result := builder.IsDockerRunning(ctx)

	// We can't assert the result as it depends on the environment
	// Just ensure it doesn't panic
	_ = result
}

func TestBuilder_Close(t *testing.T) {
	tests := []struct {
		name    string
		builder *Builder
		wantErr bool
	}{
		{
			name: "close_without_sdk",
			builder: &Builder{
				logger:     utils.NewLogger("test"),
				sdkBuilder: nil,
			},
			wantErr: false,
		},
		{
			name: "close_with_sdk",
			builder: &Builder{
				logger: utils.NewLogger("test"),
				sdkBuilder: &SDKBuilder{
					client: &MockDockerClient{},
				},
				useSDK: true,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.builder.Close()

			if (err != nil) != tt.wantErr {
				t.Errorf("Close() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGetProjectRoot(t *testing.T) {
	// Create a temporary directory structure
	tmpDir, err := os.MkdirTemp("", "test-project")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	// Create go.mod in temp dir
	goModPath := filepath.Join(tmpDir, "go.mod")
	err = os.WriteFile(goModPath, []byte("module test\n"), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// Save current directory
	originalDir, _ := os.Getwd()
	defer os.Chdir(originalDir)

	// Change to temp directory
	err = os.Chdir(tmpDir)
	if err != nil {
		t.Fatal(err)
	}

	// Test from root
	root, err := GetProjectRoot()
	if err != nil {
		t.Errorf("GetProjectRoot() error = %v", err)
	}

	// Resolve symlinks for comparison
	rootResolved, _ := filepath.EvalSymlinks(root)
	tmpDirResolved, _ := filepath.EvalSymlinks(tmpDir)

	if rootResolved != tmpDirResolved {
		t.Errorf("GetProjectRoot() = %v, want %v", root, tmpDir)
	}

	// Test from subdirectory
	subDir := filepath.Join(tmpDir, "sub")
	os.Mkdir(subDir, 0755)
	os.Chdir(subDir)

	root, err = GetProjectRoot()
	if err != nil {
		t.Errorf("GetProjectRoot() from subdir error = %v", err)
	}

	// Resolve symlinks for comparison
	rootResolved, _ = filepath.EvalSymlinks(root)

	if rootResolved != tmpDirResolved {
		t.Errorf("GetProjectRoot() from subdir = %v, want %v", root, tmpDir)
	}
}

func TestGetProjectRoot_NotFound(t *testing.T) {
	// Create a deep directory without go.mod
	tmpDir, err := os.MkdirTemp("", "test-no-root")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	deepPath := tmpDir
	for i := 0; i < 6; i++ {
		deepPath = filepath.Join(deepPath, "level")
	}
	os.MkdirAll(deepPath, 0755)

	// Save current directory
	originalDir, _ := os.Getwd()
	defer os.Chdir(originalDir)

	// Change to deep directory
	os.Chdir(deepPath)

	_, err = GetProjectRoot()
	if err == nil {
		t.Error("GetProjectRoot() expected error when go.mod not found")
	}
}

func TestRunOptions(t *testing.T) {
	// Test RunOptions struct
	opts := &RunOptions{
		Name:    "test-container",
		Env:     map[string]string{"KEY": "value"},
		Volumes: []string{"/host:/container"},
		Ports:   map[string]string{"8080": "80"},
		Network: "bridge",
		Command: []string{"sh", "-c", "echo hello"},
		Detach:  true,
	}

	// Verify fields
	if opts.Name != "test-container" {
		t.Errorf("RunOptions.Name = %v, want %v", opts.Name, "test-container")
	}

	if len(opts.Env) != 1 || opts.Env["KEY"] != "value" {
		t.Error("RunOptions.Env not set correctly")
	}

	if len(opts.Volumes) != 1 || opts.Volumes[0] != "/host:/container" {
		t.Error("RunOptions.Volumes not set correctly")
	}

	if len(opts.Ports) != 1 || opts.Ports["8080"] != "80" {
		t.Error("RunOptions.Ports not set correctly")
	}

	if opts.Network != "bridge" {
		t.Errorf("RunOptions.Network = %v, want %v", opts.Network, "bridge")
	}

	if len(opts.Command) != 3 || opts.Command[0] != "sh" {
		t.Error("RunOptions.Command not set correctly")
	}

	if !opts.Detach {
		t.Error("RunOptions.Detach should be true")
	}
}

// Test CLI fallback methods
func TestBuilder_Build_CLIFallback(t *testing.T) {
	builder := &Builder{
		logger: utils.NewLogger("test"),
		useSDK: false,
	}

	ctx := context.Background()
	// Use a non-existent Dockerfile to ensure build fails
	err := builder.Build(ctx, "Dockerfile.nonexistent", "test-cli-build:shouldnotexist", ".")

	// We expect an error (missing Dockerfile or Docker not available)
	if err == nil {
		t.Error("Build() expected error in test environment")
		// If somehow it succeeded, try to clean up
		builder.Remove(ctx, "test-cli-build:shouldnotexist")
	}
}

func TestBuilder_Tag_CLIFallback(t *testing.T) {
	builder := &Builder{
		logger: utils.NewLogger("test"),
		useSDK: false,
	}

	ctx := context.Background()
	// This will fail as docker tag won't work without real images
	err := builder.Tag(ctx, "source:latest", "target:v1.0")

	// We expect an error in test environment
	if err == nil {
		t.Error("Tag() expected error in test environment")
	}
}

func TestBuilder_Push_CLIFallback(t *testing.T) {
	builder := &Builder{
		logger: utils.NewLogger("test"),
		useSDK: false,
	}

	ctx := context.Background()
	// This will fail as docker push won't work without real images
	err := builder.Push(ctx, "test:latest")

	// We expect an error in test environment
	if err == nil {
		t.Error("Push() expected error in test environment")
	}
}

func TestBuilder_ImportToK3d(t *testing.T) {
	builder := &Builder{
		logger: utils.NewLogger("test"),
	}

	ctx := context.Background()
	// This will fail as k3d is not available in test environment
	err := builder.ImportToK3d(ctx, "test:latest", "test-cluster")

	// k3d might or might not be available
	if err == nil {
		t.Log("k3d is available - import command executed")
	} else {
		t.Log("k3d not available or import failed")
		// The error should mention k3d
		if !strings.Contains(err.Error(), "k3d") {
			t.Errorf("ImportToK3d() error should mention k3d, got: %v", err)
		}
	}
}

func TestBuilder_Run_CLIFallback(t *testing.T) {
	builder := &Builder{
		logger: utils.NewLogger("test"),
		useSDK: false,
	}

	ctx := context.Background()
	opts := &RunOptions{
		Name:    "test-container",
		Env:     map[string]string{"KEY": "value"},
		Volumes: []string{"/host:/container"},
		Ports:   map[string]string{"8080": "80"},
		Network: "bridge",
		Command: []string{"echo", "hello"},
		Detach:  true,
	}

	// This will fail as docker run won't work without real images
	_, err := builder.Run(ctx, "test:latest", opts)

	// We expect an error in test environment
	if err == nil {
		t.Error("Run() expected error in test environment")
	}
}

func TestBuilder_Stop_CLIFallback(t *testing.T) {
	builder := &Builder{
		logger: utils.NewLogger("test"),
		useSDK: false,
	}

	ctx := context.Background()
	// This will fail as there's no container to stop
	err := builder.Stop(ctx, "container-123")

	// We expect an error in test environment
	if err == nil {
		t.Error("Stop() expected error in test environment")
	}
}

func TestBuilder_Remove_CLIFallback(t *testing.T) {
	builder := &Builder{
		logger: utils.NewLogger("test"),
		useSDK: false,
	}

	ctx := context.Background()
	// This will fail as there's no container to remove
	err := builder.Remove(ctx, "container-123")

	// In some environments, Docker might be available
	if err == nil {
		t.Log("Docker is available - remove command executed")
	} else {
		t.Log("Docker not available or container doesn't exist")
	}
}

// Test SDK delegation
func TestBuilder_Build_WithSDK(t *testing.T) {
	mockClient := &MockDockerClient{}
	builder := &Builder{
		logger: utils.NewLogger("test"),
		sdkBuilder: &SDKBuilder{
			client: mockClient,
			logger: utils.NewLogger("test-sdk"),
		},
		useSDK: true,
	}

	ctx := context.Background()
	
	// Create a temp directory for build context
	tmpDir, err := os.MkdirTemp("", "build-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	// Create a test Dockerfile
	dockerfilePath := filepath.Join(tmpDir, "Dockerfile")
	if err := os.WriteFile(dockerfilePath, []byte("FROM alpine"), 0644); err != nil {
		t.Fatal(err)
	}

	err = builder.Build(ctx, dockerfilePath, "test:latest", tmpDir)
	if err != nil {
		t.Errorf("Build() with SDK error = %v", err)
	}
}

func TestBuilder_Tag_WithSDK(t *testing.T) {
	mockClient := &MockDockerClient{}
	builder := &Builder{
		logger: utils.NewLogger("test"),
		sdkBuilder: &SDKBuilder{
			client: mockClient,
			logger: utils.NewLogger("test-sdk"),
		},
		useSDK: true,
	}

	ctx := context.Background()
	err := builder.Tag(ctx, "source:latest", "target:v1.0")
	if err != nil {
		t.Errorf("Tag() with SDK error = %v", err)
	}
}

func TestBuilder_Push_WithSDK(t *testing.T) {
	mockClient := &MockDockerClient{}
	builder := &Builder{
		logger: utils.NewLogger("test"),
		sdkBuilder: &SDKBuilder{
			client: mockClient,
			logger: utils.NewLogger("test-sdk"),
		},
		useSDK: true,
	}

	ctx := context.Background()
	err := builder.Push(ctx, "test:latest")
	if err != nil {
		t.Errorf("Push() with SDK error = %v", err)
	}
}

func TestBuilder_Run_WithSDK(t *testing.T) {
	mockClient := &MockDockerClient{
		containerID: "container-123",
		logs:        "hello world",
	}
	builder := &Builder{
		logger: utils.NewLogger("test"),
		sdkBuilder: &SDKBuilder{
			client: mockClient,
			logger: utils.NewLogger("test-sdk"),
		},
		useSDK: true,
	}

	ctx := context.Background()
	opts := &RunOptions{
		Name:    "test-container",
		Command: []string{"echo", "hello"},
	}

	output, err := builder.Run(ctx, "test:latest", opts)
	if err != nil {
		t.Errorf("Run() with SDK error = %v", err)
	}
	if output != "hello world" {
		t.Errorf("Run() output = %v, want %v", output, "hello world")
	}
}

func TestBuilder_Stop_WithSDK(t *testing.T) {
	mockClient := &MockDockerClient{}
	builder := &Builder{
		logger: utils.NewLogger("test"),
		sdkBuilder: &SDKBuilder{
			client: mockClient,
			logger: utils.NewLogger("test-sdk"),
		},
		useSDK: true,
	}

	ctx := context.Background()
	err := builder.Stop(ctx, "container-123")
	if err != nil {
		t.Errorf("Stop() with SDK error = %v", err)
	}
}

func TestBuilder_Remove_WithSDK(t *testing.T) {
	mockClient := &MockDockerClient{}
	builder := &Builder{
		logger: utils.NewLogger("test"),
		sdkBuilder: &SDKBuilder{
			client: mockClient,
			logger: utils.NewLogger("test-sdk"),
		},
		useSDK: true,
	}

	ctx := context.Background()
	err := builder.Remove(ctx, "container-123")
	if err != nil {
		t.Errorf("Remove() with SDK error = %v", err)
	}
}

func TestBuilder_IsDockerRunning_WithSDK(t *testing.T) {
	tests := []struct {
		name       string
		mockClient *MockDockerClient
		want       bool
	}{
		{
			name:       "docker_running",
			mockClient: &MockDockerClient{},
			want:       true,
		},
		{
			name: "docker_not_running",
			mockClient: &MockDockerClient{
				pingErr: errors.New("not running"),
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := &Builder{
				logger: utils.NewLogger("test"),
				sdkBuilder: &SDKBuilder{
					client: tt.mockClient,
				},
				useSDK: true,
			}

			ctx := context.Background()
			got := builder.IsDockerRunning(ctx)
			if got != tt.want {
				t.Errorf("IsDockerRunning() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Benchmark tests
func BenchmarkNewBuilder(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = NewBuilder()
	}
}

func BenchmarkGetProjectRoot(b *testing.B) {
	// Create a temp dir with go.mod
	tmpDir, _ := os.MkdirTemp("", "bench-project")
	defer os.RemoveAll(tmpDir)

	goModPath := filepath.Join(tmpDir, "go.mod")
	os.WriteFile(goModPath, []byte("module bench\n"), 0644)

	originalDir, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(originalDir)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = GetProjectRoot()
	}
}
