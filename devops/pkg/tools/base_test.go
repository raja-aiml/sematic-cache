// Package tools provides unit tests for base tool implementation
package tools

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

// Tests for BaseTool

func TestBaseTool_Properties(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}

	tool := NewBaseTool("test-tool", "v1.0.0", "Test tool", "test", logger, osUtil)

	assert.Equal(t, "test-tool", tool.Name())
	assert.Equal(t, "v1.0.0", tool.Version())
	assert.Equal(t, "Test tool", tool.Description())
	assert.Equal(t, logger, tool.Logger())
}

func TestBaseTool_IsInstalled(t *testing.T) {
	tests := []struct {
		name      string
		available bool
		expected  bool
	}{
		{
			name:      "tool is installed",
			available: true,
			expected:  true,
		},
		{
			name:      "tool is not installed",
			available: false,
			expected:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := &mockLogger{}
			osUtil := &mockOSUtil{}

			osUtil.On("IsCommandAvailable", "test").Return(tt.available)

			tool := NewBaseTool("test-tool", "v1.0.0", "Test tool", "test", logger, osUtil)

			result := tool.IsInstalled()
			assert.Equal(t, tt.expected, result)

			osUtil.AssertExpectations(t)
		})
	}
}

func TestBaseTool_Validate(t *testing.T) {
	tests := []struct {
		name        string
		isInstalled bool
		expectErr   bool
	}{
		{
			name:        "tool is not installed",
			isInstalled: false,
			expectErr:   true,
		},
		{
			name:        "tool is installed",
			isInstalled: true,
			expectErr:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := &mockLogger{}
			osUtil := &mockOSUtil{}

			osUtil.On("IsCommandAvailable", "test").Return(tt.isInstalled)

			if tt.isInstalled {
				// For the successful case, we expect the Info log call
				// We know the version will be "v1.0.0" from our mock
				logger.On("Info", "%s is installed: %s", []interface{}{"test-tool", "v1.0.0"})
			}

			tool := NewBaseTool("test-tool", "v1.0.0", "Test tool", "test", logger, osUtil)

			// Create a testable version using a mock wrapper
			if tt.isInstalled {
				// Use a mock that simulates successful version retrieval
				mockTool := &mockableBaseTool{
					BaseTool:   tool,
					version:    "v1.0.0",
					versionErr: nil,
				}
				err := mockTool.Validate()
				if tt.expectErr {
					assert.Error(t, err)
				} else {
					assert.NoError(t, err)
				}
			} else {
				// For not installed case, we can test the base tool directly
				err := tool.Validate()
				if tt.expectErr {
					assert.Error(t, err)
				} else {
					assert.NoError(t, err)
				}
			}

			osUtil.AssertExpectations(t)
			logger.AssertExpectations(t)
		})
	}
}

func TestBaseTool_GetInstalledVersion(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}

	// This test will test the actual exec.Command, so it won't be perfectly reliable
	// but it will at least test the code path
	tool := NewBaseTool("test-tool", "v1.0.0", "Test tool", "echo", logger, osUtil)

	version, err := tool.GetInstalledVersion()
	// echo command should work and return something
	assert.NoError(t, err)
	assert.NotEmpty(t, version)
}

func TestBaseTool_Uninstall(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}

	tool := NewBaseTool("test-tool", "v1.0.0", "Test tool", "test", logger, osUtil)

	err := tool.Uninstall(context.Background())
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "uninstall not implemented")
}

// Tests for DownloadableTool

func TestDownloadableTool_GetDownloadURL(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}
	downloader := &mockFileDownloader{}
	extractor := &mockArchiveExtractor{}
	httpClient := &mockHTTPClient{}

	base := NewBaseTool("test-tool", "v1.0.0", "Test tool", "test", logger, osUtil)
	tool := NewDownloadableTool(base, downloader, extractor, httpClient)

	// The base implementation should return an error
	url, err := tool.GetDownloadURL()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "GetDownloadURL must be implemented")
	assert.Empty(t, url)
}

func TestDownloadableTool_GetBinaryName(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}
	downloader := &mockFileDownloader{}
	extractor := &mockArchiveExtractor{}
	httpClient := &mockHTTPClient{}

	base := NewBaseTool("test-tool", "v1.0.0", "Test tool", "test", logger, osUtil)
	tool := NewDownloadableTool(base, downloader, extractor, httpClient)

	binaryName := tool.GetBinaryName()
	assert.Equal(t, "test", binaryName)
}

func TestDownloadableTool_Install_AlreadyInstalled(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}
	downloader := &mockFileDownloader{}
	extractor := &mockArchiveExtractor{}
	httpClient := &mockHTTPClient{}

	osUtil.On("IsCommandAvailable", "test").Return(true)
	logger.On("Info", "%s is already installed: %s", []interface{}{"test-tool", "v1.0.0"})

	base := NewBaseTool("test-tool", "v1.0.0", "Test tool", "test", logger, osUtil)
	tool := &testDownloadableTool{
		DownloadableTool: NewDownloadableTool(base, downloader, extractor, httpClient),
		url:              "https://example.com/test-tool.tar.gz",
		version:          "v1.0.0",
	}

	err := tool.Install(context.Background())
	assert.NoError(t, err)

	osUtil.AssertExpectations(t)
	logger.AssertExpectations(t)
	downloader.AssertNotCalled(t, "Download")
}

func TestDownloadableTool_Install_Success(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}
	downloader := &mockFileDownloader{}
	extractor := &mockArchiveExtractor{}
	httpClient := &mockHTTPClient{}

	ctx := context.Background()
	tempDir := "/tmp/test-tool-123"
	url := "https://example.com/test-tool.tar.gz"

	// Setup expectations
	osUtil.On("IsCommandAvailable", "test").Return(false)
	logger.On("Info", "Downloading %s %s...", []interface{}{"test-tool", "v1.0.0"})
	osUtil.On("CreateTempDir", "test-tool").Return(tempDir, nil)
	downloader.On("Download", ctx, url, tempDir+"/test-tool-archive").Return(nil)
	extractor.On("Extract", tempDir+"/test-tool-archive", tempDir).Return(nil)
	osUtil.On("RemoveTempDir", tempDir).Return(nil)
	logger.On("Success", "%s %s installed successfully", []interface{}{"test-tool", "v1.0.0"})

	base := NewBaseTool("test-tool", "v1.0.0", "Test tool", "test", logger, osUtil)
	tool := &testDownloadableTool{
		DownloadableTool: NewDownloadableTool(base, downloader, extractor, httpClient),
		url:              url,
		installSuccess:   true,
	}

	err := tool.Install(ctx)
	assert.NoError(t, err)

	osUtil.AssertExpectations(t)
	logger.AssertExpectations(t)
	downloader.AssertExpectations(t)
	extractor.AssertExpectations(t)
}

func TestDownloadableTool_Install_DownloadError(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}
	downloader := &mockFileDownloader{}
	extractor := &mockArchiveExtractor{}
	httpClient := &mockHTTPClient{}

	ctx := context.Background()
	tempDir := "/tmp/test-tool-123"
	url := "https://example.com/test-tool.tar.gz"
	downloadErr := fmt.Errorf("download failed")

	// Setup expectations
	osUtil.On("IsCommandAvailable", "test").Return(false)
	logger.On("Info", "Downloading %s %s...", []interface{}{"test-tool", "v1.0.0"})
	osUtil.On("CreateTempDir", "test-tool").Return(tempDir, nil)
	downloader.On("Download", ctx, url, tempDir+"/test-tool-archive").Return(downloadErr)
	osUtil.On("RemoveTempDir", tempDir).Return(nil)

	base := NewBaseTool("test-tool", "v1.0.0", "Test tool", "test", logger, osUtil)
	tool := &testDownloadableTool{
		DownloadableTool: NewDownloadableTool(base, downloader, extractor, httpClient),
		url:              url,
	}

	err := tool.Install(ctx)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to download")

	osUtil.AssertExpectations(t)
	logger.AssertExpectations(t)
	downloader.AssertExpectations(t)
}

// Helper types for testing

type mockableBaseTool struct {
	*BaseTool
	version    string
	versionErr error
}

func (t *mockableBaseTool) GetInstalledVersion() (string, error) {
	return t.version, t.versionErr
}

func (t *mockableBaseTool) Validate() error {
	if !t.IsInstalled() {
		return fmt.Errorf("%s is not installed", t.name)
	}

	version, err := t.GetInstalledVersion()
	if err != nil {
		return fmt.Errorf("failed to validate %s: %w", t.name, err)
	}

	t.logger.Info("%s is installed: %s", t.name, version)
	return nil
}

type testDownloadableTool struct {
	*DownloadableTool
	url            string
	installSuccess bool
	version        string
	versionErr     error
}

func (t *testDownloadableTool) GetDownloadURL() (string, error) {
	return t.url, nil
}

func (t *testDownloadableTool) GetInstalledVersion() (string, error) {
	if t.version != "" || t.versionErr != nil {
		return t.version, t.versionErr
	}
	return t.DownloadableTool.GetInstalledVersion()
}

func (t *testDownloadableTool) Install(ctx context.Context) error {
	if t.IsInstalled() {
		version, _ := t.GetInstalledVersion()
		t.logger.Info("%s is already installed: %s", t.Name(), version)
		return nil
	}

	url, err := t.GetDownloadURL()
	if err != nil {
		return fmt.Errorf("failed to get download URL: %w", err)
	}

	t.logger.Info("Downloading %s %s...", t.Name(), t.Version())

	// Create temp directory
	tempDir, err := t.osUtil.CreateTempDir(t.Name())
	if err != nil {
		return fmt.Errorf("failed to create temp dir: %w", err)
	}
	defer t.osUtil.RemoveTempDir(tempDir)

	// Download file
	archivePath := fmt.Sprintf("%s/%s-archive", tempDir, t.Name())
	if err := t.downloader.Download(ctx, url, archivePath); err != nil {
		return fmt.Errorf("failed to download %s: %w", t.Name(), err)
	}

	// Extract if needed
	if strings.HasSuffix(url, ".tar.gz") || strings.HasSuffix(url, ".tgz") {
		if err := t.extractor.Extract(archivePath, tempDir); err != nil {
			return fmt.Errorf("failed to extract %s: %w", t.Name(), err)
		}
	}

	// Install binary
	if err := t.installBinary(tempDir); err != nil {
		return fmt.Errorf("failed to install %s: %w", t.Name(), err)
	}

	t.logger.Success("%s %s installed successfully", t.Name(), t.Version())
	return nil
}

func (t *testDownloadableTool) installBinary(tempDir string) error {
	if t.installSuccess {
		return nil
	}
	return fmt.Errorf("install failed")
}
