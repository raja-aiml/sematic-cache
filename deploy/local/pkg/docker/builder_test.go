package docker

import (
	"context"
	"os"
	"path/filepath"
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
	// This will fail as docker build won't work in test environment
	err := builder.Build(ctx, "Dockerfile", "test:latest", ".")

	// We expect an error in test environment
	if err == nil {
		t.Error("Build() expected error in test environment")
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

	if err == nil {
		t.Error("ImportToK3d() expected error in test environment")
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

	// We expect an error in test environment
	if err == nil {
		t.Error("Remove() expected error in test environment")
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
