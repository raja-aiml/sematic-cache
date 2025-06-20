package docker

import (
	"archive/tar"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/api/types/network"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

// MockDockerClient implements the Docker client interface for testing
type MockDockerClient struct {
	pingErr          error
	imageBuildErr    error
	imageTagErr      error
	imagePushErr     error
	imageListErr     error
	containerCreateErr error
	containerStartErr  error
	containerStopErr   error
	containerRemoveErr error
	containerWaitErr   error
	containerLogsErr   error
	buildResponse      *types.ImageBuildResponse
	pushResponse       io.ReadCloser
	images            []image.Summary
	containerID       string
	logs              string
}

func (m *MockDockerClient) Ping(ctx context.Context) (types.Ping, error) {
	return types.Ping{}, m.pingErr
}

func (m *MockDockerClient) ImageBuild(ctx context.Context, buildContext io.Reader, options types.ImageBuildOptions) (types.ImageBuildResponse, error) {
	if m.imageBuildErr != nil {
		return types.ImageBuildResponse{}, m.imageBuildErr
	}
	if m.buildResponse != nil {
		return *m.buildResponse, nil
	}
	// Return a successful build response
	successMsg := `{"stream":"Successfully built\n"}`
	return types.ImageBuildResponse{
		Body: io.NopCloser(strings.NewReader(successMsg)),
	}, nil
}

func (m *MockDockerClient) ImageTag(ctx context.Context, source, target string) error {
	return m.imageTagErr
}

func (m *MockDockerClient) ImagePush(ctx context.Context, image string, options image.PushOptions) (io.ReadCloser, error) {
	if m.imagePushErr != nil {
		return nil, m.imagePushErr
	}
	if m.pushResponse != nil {
		return m.pushResponse, nil
	}
	// Return a successful push response
	successMsg := `{"status":"Push complete"}`
	return io.NopCloser(strings.NewReader(successMsg)), nil
}

func (m *MockDockerClient) ImageList(ctx context.Context, options image.ListOptions) ([]image.Summary, error) {
	if m.imageListErr != nil {
		return nil, m.imageListErr
	}
	return m.images, nil
}

func (m *MockDockerClient) ContainerCreate(ctx context.Context, config *container.Config, hostConfig *container.HostConfig, networkingConfig *network.NetworkingConfig, platform *specs.Platform, containerName string) (container.CreateResponse, error) {
	if m.containerCreateErr != nil {
		return container.CreateResponse{}, m.containerCreateErr
	}
	return container.CreateResponse{
		ID: m.containerID,
	}, nil
}

func (m *MockDockerClient) ContainerStart(ctx context.Context, containerID string, options container.StartOptions) error {
	return m.containerStartErr
}

func (m *MockDockerClient) ContainerStop(ctx context.Context, containerID string, options container.StopOptions) error {
	return m.containerStopErr
}

func (m *MockDockerClient) ContainerRemove(ctx context.Context, containerID string, options container.RemoveOptions) error {
	return m.containerRemoveErr
}

func (m *MockDockerClient) ContainerWait(ctx context.Context, containerID string, condition container.WaitCondition) (<-chan container.WaitResponse, <-chan error) {
	statusCh := make(chan container.WaitResponse, 1)
	errCh := make(chan error, 1)
	
	go func() {
		if m.containerWaitErr != nil {
			errCh <- m.containerWaitErr
		} else {
			statusCh <- container.WaitResponse{StatusCode: 0}
		}
		close(statusCh)
		close(errCh)
	}()
	
	return statusCh, errCh
}

func (m *MockDockerClient) ContainerLogs(ctx context.Context, container string, options container.LogsOptions) (io.ReadCloser, error) {
	if m.containerLogsErr != nil {
		return nil, m.containerLogsErr
	}
	
	// Create a proper multiplexed stream
	var buf bytes.Buffer
	// Write stdout header (stream type 1, then size as big-endian uint32)
	msgLen := len(m.logs)
	buf.Write([]byte{1, 0, 0, 0, 
		byte(msgLen >> 24), byte(msgLen >> 16), byte(msgLen >> 8), byte(msgLen)})
	buf.WriteString(m.logs)
	
	return io.NopCloser(&buf), nil
}

func (m *MockDockerClient) Close() error {
	return nil
}

func TestNewSDKBuilder(t *testing.T) {
	// Test actual NewSDKBuilder function
	builder, err := NewSDKBuilder()
	
	// The function will either succeed (if Docker is available)
	// or fail (if Docker is not available)
	if err != nil {
		t.Logf("NewSDKBuilder() failed as expected without Docker: %v", err)
		// Ensure error message is descriptive
		if !strings.Contains(err.Error(), "docker client") {
			t.Errorf("NewSDKBuilder() error should mention docker client, got: %v", err)
		}
	} else {
		t.Log("NewSDKBuilder() succeeded with Docker available")
		// If successful, builder should not be nil
		if builder == nil {
			t.Error("NewSDKBuilder() returned nil builder with nil error")
		} else {
			// Clean up
			builder.Close()
		}
	}
}

func TestSDKBuilder_Close(t *testing.T) {
	builder := &SDKBuilder{
		client: &MockDockerClient{},
		logger: nil,
	}
	
	err := builder.Close()
	if err != nil {
		t.Errorf("Close() error = %v", err)
	}
	
	// Test with nil client
	builder.client = nil
	err = builder.Close()
	if err != nil {
		t.Errorf("Close() with nil client error = %v", err)
	}
}

func TestSDKBuilder_Build(t *testing.T) {
	// Create a temporary directory for testing
	tmpDir, err := os.MkdirTemp("", "docker-build-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)
	
	// Create a test Dockerfile
	dockerfilePath := filepath.Join(tmpDir, "Dockerfile")
	dockerfileContent := "FROM alpine\nRUN echo hello"
	if err := os.WriteFile(dockerfilePath, []byte(dockerfileContent), 0644); err != nil {
		t.Fatal(err)
	}
	
	// Create a test file in context
	testFile := filepath.Join(tmpDir, "test.txt")
	if err := os.WriteFile(testFile, []byte("test content"), 0644); err != nil {
		t.Fatal(err)
	}
	
	tests := []struct {
		name           string
		dockerfilePath string
		imageName      string
		buildContext   string
		mockClient     *MockDockerClient
		wantErr        bool
		errContains    string
	}{
		{
			name:           "successful_build",
			dockerfilePath: dockerfilePath,
			imageName:      "test:latest",
			buildContext:   tmpDir,
			mockClient:     &MockDockerClient{},
			wantErr:        false,
		},
		{
			name:           "build_api_error",
			dockerfilePath: dockerfilePath,
			imageName:      "test:latest",
			buildContext:   tmpDir,
			mockClient: &MockDockerClient{
				imageBuildErr: errors.New("API error"),
			},
			wantErr:     true,
			errContains: "docker build failed",
		},
		{
			name:           "build_output_error",
			dockerfilePath: dockerfilePath,
			imageName:      "test:latest",
			buildContext:   tmpDir,
			mockClient: &MockDockerClient{
				buildResponse: &types.ImageBuildResponse{
					Body: io.NopCloser(strings.NewReader(`{"errorDetail":{"message":"build failed"}}`)),
				},
			},
			wantErr:     true,
			errContains: "build error",
		},
		{
			name:           "invalid_context",
			dockerfilePath: dockerfilePath,
			imageName:      "test:latest",
			buildContext:   "/nonexistent/path",
			mockClient:     &MockDockerClient{},
			wantErr:        true,
			errContains:    "failed to create build context",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := &SDKBuilder{
				client: tt.mockClient,
				logger: utils.NewLogger("test"),
			}
			
			ctx := context.Background()
			err := builder.Build(ctx, tt.dockerfilePath, tt.imageName, tt.buildContext)
			
			if (err != nil) != tt.wantErr {
				t.Errorf("Build() error = %v, wantErr %v", err, tt.wantErr)
			}
			
			if err != nil && tt.errContains != "" && !strings.Contains(err.Error(), tt.errContains) {
				t.Errorf("Build() error = %v, want error containing %v", err, tt.errContains)
			}
		})
	}
}

func TestSDKBuilder_Tag(t *testing.T) {
	tests := []struct {
		name        string
		sourceImage string
		targetImage string
		mockClient  *MockDockerClient
		wantErr     bool
	}{
		{
			name:        "successful_tag",
			sourceImage: "source:latest",
			targetImage: "target:v1.0",
			mockClient:  &MockDockerClient{},
			wantErr:     false,
		},
		{
			name:        "tag_error",
			sourceImage: "source:latest",
			targetImage: "target:v1.0",
			mockClient: &MockDockerClient{
				imageTagErr: errors.New("tag failed"),
			},
			wantErr: true,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := &SDKBuilder{
				client: tt.mockClient,
				logger: utils.NewLogger("test"),
			}
			
			ctx := context.Background()
			err := builder.Tag(ctx, tt.sourceImage, tt.targetImage)
			
			if (err != nil) != tt.wantErr {
				t.Errorf("Tag() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestSDKBuilder_Push(t *testing.T) {
	tests := []struct {
		name        string
		imageName   string
		mockClient  *MockDockerClient
		wantErr     bool
		errContains string
	}{
		{
			name:       "successful_push",
			imageName:  "test:latest",
			mockClient: &MockDockerClient{},
			wantErr:    false,
		},
		{
			name:      "push_api_error",
			imageName: "test:latest",
			mockClient: &MockDockerClient{
				imagePushErr: errors.New("push failed"),
			},
			wantErr:     true,
			errContains: "docker push failed",
		},
		{
			name:      "push_output_error",
			imageName: "test:latest",
			mockClient: &MockDockerClient{
				pushResponse: io.NopCloser(strings.NewReader(`{"errorDetail":{"message":"push error"}}`)),
			},
			wantErr:     true,
			errContains: "push error",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := &SDKBuilder{
				client: tt.mockClient,
				logger: utils.NewLogger("test"),
			}
			
			ctx := context.Background()
			err := builder.Push(ctx, tt.imageName)
			
			if (err != nil) != tt.wantErr {
				t.Errorf("Push() error = %v, wantErr %v", err, tt.wantErr)
			}
			
			if err != nil && tt.errContains != "" && !strings.Contains(err.Error(), tt.errContains) {
				t.Errorf("Push() error = %v, want error containing %v", err, tt.errContains)
			}
		})
	}
}

func TestSDKBuilder_Run(t *testing.T) {
	tests := []struct {
		name        string
		imageName   string
		opts        *RunOptions
		mockClient  *MockDockerClient
		wantOutput  string
		wantErr     bool
		errContains string
	}{
		{
			name:      "successful_run_attached",
			imageName: "test:latest",
			opts:      &RunOptions{},
			mockClient: &MockDockerClient{
				containerID: "container-123",
				logs:        "hello",
			},
			wantOutput: "hello",
			wantErr:    false,
		},
		{
			name:      "successful_run_detached",
			imageName: "test:latest",
			opts: &RunOptions{
				Detach: true,
			},
			mockClient: &MockDockerClient{
				containerID: "container-123",
			},
			wantOutput: "container-123",
			wantErr:    false,
		},
		{
			name:      "run_with_all_options",
			imageName: "test:latest",
			opts: &RunOptions{
				Name:    "test-container",
				Env:     map[string]string{"KEY": "value"},
				Volumes: []string{"/host:/container"},
				Ports:   map[string]string{"8080": "80"},
				Network: "bridge",
				Command: []string{"echo", "test"},
			},
			mockClient: &MockDockerClient{
				containerID: "container-123",
				logs:        "test",
			},
			wantOutput: "test",
			wantErr:    false,
		},
		{
			name:      "container_create_error",
			imageName: "test:latest",
			opts:      &RunOptions{},
			mockClient: &MockDockerClient{
				containerCreateErr: errors.New("create failed"),
			},
			wantErr:     true,
			errContains: "container create failed",
		},
		{
			name:      "container_start_error",
			imageName: "test:latest",
			opts:      &RunOptions{},
			mockClient: &MockDockerClient{
				containerID:       "container-123",
				containerStartErr: errors.New("start failed"),
			},
			wantErr:     true,
			errContains: "container start failed",
		},
		{
			name:      "container_wait_error",
			imageName: "test:latest",
			opts:      &RunOptions{},
			mockClient: &MockDockerClient{
				containerID:      "container-123",
				containerWaitErr: errors.New("wait failed"),
			},
			wantErr:     true,
			errContains: "container wait failed",
		},
		{
			name:      "container_logs_error",
			imageName: "test:latest",
			opts:      &RunOptions{},
			mockClient: &MockDockerClient{
				containerID:      "container-123",
				containerLogsErr: errors.New("logs failed"),
			},
			wantErr:     true,
			errContains: "failed to get container logs",
		},
		{
			name:      "invalid_port",
			imageName: "test:latest",
			opts: &RunOptions{
				Ports: map[string]string{"8080": "80"},
			},
			mockClient: &MockDockerClient{
				containerID: "container-123",
				logs:        "",
			},
			wantOutput: "",
			wantErr:    false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := &SDKBuilder{
				client: tt.mockClient,
				logger: utils.NewLogger("test"),
			}
			
			ctx := context.Background()
			output, err := builder.Run(ctx, tt.imageName, tt.opts)
			
			if (err != nil) != tt.wantErr {
				t.Errorf("Run() error = %v, wantErr %v", err, tt.wantErr)
			}
			
			if err != nil && tt.errContains != "" && !strings.Contains(err.Error(), tt.errContains) {
				t.Errorf("Run() error = %v, want error containing %v", err, tt.errContains)
			}
			
			if err == nil && output != tt.wantOutput {
				t.Errorf("Run() output = %v, want %v", output, tt.wantOutput)
			}
		})
	}
}

func TestSDKBuilder_Stop(t *testing.T) {
	tests := []struct {
		name        string
		containerID string
		mockClient  *MockDockerClient
		wantErr     bool
	}{
		{
			name:        "successful_stop",
			containerID: "container-123",
			mockClient:  &MockDockerClient{},
			wantErr:     false,
		},
		{
			name:        "stop_error",
			containerID: "container-123",
			mockClient: &MockDockerClient{
				containerStopErr: errors.New("stop failed"),
			},
			wantErr: true,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := &SDKBuilder{
				client: tt.mockClient,
				logger: utils.NewLogger("test"),
			}
			
			ctx := context.Background()
			err := builder.Stop(ctx, tt.containerID)
			
			if (err != nil) != tt.wantErr {
				t.Errorf("Stop() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestSDKBuilder_Remove(t *testing.T) {
	tests := []struct {
		name        string
		containerID string
		mockClient  *MockDockerClient
		wantErr     bool
	}{
		{
			name:        "successful_remove",
			containerID: "container-123",
			mockClient:  &MockDockerClient{},
			wantErr:     false,
		},
		{
			name:        "remove_error",
			containerID: "container-123",
			mockClient: &MockDockerClient{
				containerRemoveErr: errors.New("remove failed"),
			},
			wantErr: true,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := &SDKBuilder{
				client: tt.mockClient,
				logger: utils.NewLogger("test"),
			}
			
			ctx := context.Background()
			err := builder.Remove(ctx, tt.containerID)
			
			if (err != nil) != tt.wantErr {
				t.Errorf("Remove() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestSDKBuilder_IsDockerRunning(t *testing.T) {
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
				pingErr: errors.New("cannot connect"),
			},
			want: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := &SDKBuilder{
				client: tt.mockClient,
			}
			
			ctx := context.Background()
			got := builder.IsDockerRunning(ctx)
			
			if got != tt.want {
				t.Errorf("IsDockerRunning() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSDKBuilder_ListImages(t *testing.T) {
	tests := []struct {
		name       string
		mockClient *MockDockerClient
		wantImages []image.Summary
		wantErr    bool
	}{
		{
			name: "successful_list",
			mockClient: &MockDockerClient{
				images: []image.Summary{
					{ID: "image1", RepoTags: []string{"test:latest"}},
					{ID: "image2", RepoTags: []string{"test:v1.0"}},
				},
			},
			wantImages: []image.Summary{
				{ID: "image1", RepoTags: []string{"test:latest"}},
				{ID: "image2", RepoTags: []string{"test:v1.0"}},
			},
			wantErr: false,
		},
		{
			name: "list_error",
			mockClient: &MockDockerClient{
				imageListErr: errors.New("list failed"),
			},
			wantErr: true,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := &SDKBuilder{
				client: tt.mockClient,
			}
			
			ctx := context.Background()
			images, err := builder.ListImages(ctx)
			
			if (err != nil) != tt.wantErr {
				t.Errorf("ListImages() error = %v, wantErr %v", err, tt.wantErr)
			}
			
			if !tt.wantErr && len(images) != len(tt.wantImages) {
				t.Errorf("ListImages() returned %d images, want %d", len(images), len(tt.wantImages))
			}
		})
	}
}

func TestCreateTarArchive(t *testing.T) {
	// Create a temporary directory
	tmpDir, err := os.MkdirTemp("", "tar-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)
	
	// Create test files
	testFile := filepath.Join(tmpDir, "test.txt")
	if err := os.WriteFile(testFile, []byte("test content"), 0644); err != nil {
		t.Fatal(err)
	}
	
	// Create .git directory (should be skipped)
	gitDir := filepath.Join(tmpDir, ".git")
	if err := os.Mkdir(gitDir, 0755); err != nil {
		t.Fatal(err)
	}
	gitFile := filepath.Join(gitDir, "config")
	if err := os.WriteFile(gitFile, []byte("git config"), 0644); err != nil {
		t.Fatal(err)
	}
	
	// Create node_modules directory (should be skipped)
	nodeDir := filepath.Join(tmpDir, "node_modules")
	if err := os.Mkdir(nodeDir, 0755); err != nil {
		t.Fatal(err)
	}
	
	// Create subdirectory
	subDir := filepath.Join(tmpDir, "subdir")
	if err := os.Mkdir(subDir, 0755); err != nil {
		t.Fatal(err)
	}
	subFile := filepath.Join(subDir, "sub.txt")
	if err := os.WriteFile(subFile, []byte("sub content"), 0644); err != nil {
		t.Fatal(err)
	}
	
	// Create tar archive
	reader, err := createTarArchive(tmpDir, "Dockerfile")
	if err != nil {
		t.Fatalf("createTarArchive() error = %v", err)
	}
	
	// Verify tar contents
	tr := tar.NewReader(reader)
	fileCount := 0
	foundTestFile := false
	foundSubFile := false
	
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("tar.Next() error = %v", err)
		}
		
		fileCount++
		
		// Check that .git and node_modules are not included
		if strings.Contains(header.Name, ".git") {
			t.Errorf("tar contains .git file: %s", header.Name)
		}
		if strings.Contains(header.Name, "node_modules") {
			t.Errorf("tar contains node_modules file: %s", header.Name)
		}
		
		// Check expected files
		if header.Name == "test.txt" {
			foundTestFile = true
		}
		if header.Name == "subdir/sub.txt" {
			foundSubFile = true
		}
	}
	
	if !foundTestFile {
		t.Error("tar archive missing test.txt")
	}
	if !foundSubFile {
		t.Error("tar archive missing subdir/sub.txt")
	}
	
	// Test with non-existent directory
	_, err = createTarArchive("/nonexistent/path", "Dockerfile")
	if err == nil {
		t.Error("createTarArchive() expected error for non-existent path")
	}
	
	// Test with file that can't be read
	unreadableDir := filepath.Join(tmpDir, "unreadable")
	if err := os.Mkdir(unreadableDir, 0755); err != nil {
		t.Fatal(err)
	}
	unreadableFile := filepath.Join(unreadableDir, "noperm.txt")
	if err := os.WriteFile(unreadableFile, []byte("content"), 0644); err != nil {
		t.Fatal(err)
	}
	// Make file unreadable (but keep directory readable for traversal)
	if err := os.Chmod(unreadableFile, 0000); err != nil {
		t.Fatal(err)
	}
	defer os.Chmod(unreadableFile, 0644) // Restore permissions for cleanup
	
	// This should fail when trying to read the file
	_, err = createTarArchive(unreadableDir, "Dockerfile")
	if err == nil {
		t.Error("createTarArchive() expected error for unreadable file")
	}
}

func TestProcessBuildOutput(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
		errMsg  string
	}{
		{
			name:    "successful_build",
			input:   `{"stream":"Step 1/2 : FROM alpine\n"}{"stream":"Successfully built\n"}`,
			wantErr: false,
		},
		{
			name:    "build_error",
			input:   `{"errorDetail":{"message":"build failed"}}`,
			wantErr: true,
			errMsg:  "build error: build failed",
		},
		{
			name:    "invalid_json",
			input:   `invalid json`,
			wantErr: true,
		},
		{
			name:    "empty_input",
			input:   "",
			wantErr: false,
		},
		{
			name:    "mixed_output",
			input:   `{"stream":"Building...\n"}{"errorDetail":{"message":"error occurred"}}`,
			wantErr: true,
			errMsg:  "build error: error occurred",
		},
		{
			name:    "error_without_message",
			input:   `{"errorDetail":{}}`,
			wantErr: false,
		},
	}
	
	logger := utils.NewLogger("test")
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			err := processBuildOutput(reader, logger)
			
			if (err != nil) != tt.wantErr {
				t.Errorf("processBuildOutput() error = %v, wantErr %v", err, tt.wantErr)
			}
			
			if err != nil && tt.errMsg != "" && err.Error() != tt.errMsg {
				t.Errorf("processBuildOutput() error = %v, want %v", err, tt.errMsg)
			}
		})
	}
}

func TestProcessPushOutput(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
		errMsg  string
	}{
		{
			name:    "successful_push",
			input:   `{"status":"Pushing layer"}{"status":"Push complete"}`,
			wantErr: false,
		},
		{
			name:    "push_error",
			input:   `{"errorDetail":{"message":"push failed"}}`,
			wantErr: true,
			errMsg:  "push error: push failed",
		},
		{
			name:    "invalid_json",
			input:   `invalid json`,
			wantErr: true,
		},
		{
			name:    "empty_input",
			input:   "",
			wantErr: false,
		},
		{
			name:    "mixed_push_output",
			input:   `{"status":"Pushing..."}{"errorDetail":{"message":"auth failed"}}`,
			wantErr: true,
			errMsg:  "push error: auth failed",
		},
		{
			name:    "push_error_without_message",
			input:   `{"errorDetail":{}}`,
			wantErr: false,
		},
	}
	
	logger := utils.NewLogger("test")
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := io.NopCloser(strings.NewReader(tt.input))
			err := processPushOutput(reader, logger)
			
			if (err != nil) != tt.wantErr {
				t.Errorf("processPushOutput() error = %v, wantErr %v", err, tt.wantErr)
			}
			
			if err != nil && tt.errMsg != "" && err.Error() != tt.errMsg {
				t.Errorf("processPushOutput() error = %v, want %v", err, tt.errMsg)
			}
		})
	}
}

// Benchmark tests
func BenchmarkCreateTarArchive(b *testing.B) {
	// Create a temporary directory with some files
	tmpDir, _ := os.MkdirTemp("", "bench-tar")
	defer os.RemoveAll(tmpDir)
	
	// Create some test files
	for i := 0; i < 10; i++ {
		filename := filepath.Join(tmpDir, fmt.Sprintf("file%d.txt", i))
		os.WriteFile(filename, []byte("test content"), 0644)
	}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		_, _ = createTarArchive(tmpDir, "Dockerfile")
	}
}

func BenchmarkProcessBuildOutput(b *testing.B) {
	input := `{"stream":"Step 1/2 : FROM alpine\n"}{"stream":"Successfully built\n"}`
	logger := utils.NewLogger("bench")
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		reader := strings.NewReader(input)
		_ = processBuildOutput(reader, logger)
	}
}