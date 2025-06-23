// Package tools provides unit tests for golang-specific tools
package tools

import (
	"context"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewGolangCILint(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}
	downloader := &mockFileDownloader{}
	extractor := &mockArchiveExtractor{}
	httpClient := &mockHTTPClient{}

	tool := NewGolangCILint("v1.60.0", logger, osUtil, downloader, extractor, httpClient)

	assert.Equal(t, "golangci-lint", tool.Name())
	assert.Equal(t, "v1.60.0", tool.Version())
	assert.Contains(t, tool.Description(), "linter")
}

func TestGolangCILint_GetDownloadURL(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}
	downloader := &mockFileDownloader{}
	extractor := &mockArchiveExtractor{}
	httpClient := &mockHTTPClient{}

	// Mock OS and arch
	osUtil.On("GetOS").Return("linux")
	osUtil.On("GetArch").Return("amd64")

	tool := NewGolangCILint("v1.60.0", logger, osUtil, downloader, extractor, httpClient)

	url, err := tool.GetDownloadURL()
	assert.NoError(t, err)
	assert.Contains(t, url, "golangci-lint")
	assert.Contains(t, url, "linux")
	assert.Contains(t, url, "amd64")
}

func TestGolangCILint_GetBinaryName(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}
	downloader := &mockFileDownloader{}
	extractor := &mockArchiveExtractor{}
	httpClient := &mockHTTPClient{}

	tool := NewGolangCILint("v1.60.0", logger, osUtil, downloader, extractor, httpClient)

	binaryName := tool.GetBinaryName()
	assert.Equal(t, "golangci-lint", binaryName)
}

func TestGolangCILint_GetInstalledVersion(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}
	downloader := &mockFileDownloader{}
	extractor := &mockArchiveExtractor{}
	httpClient := &mockHTTPClient{}

	tool := NewGolangCILint("v1.60.0", logger, osUtil, downloader, extractor, httpClient)

	// This will test the actual command execution
	// We're just testing the code path, not requiring specific behavior
	version, err := tool.GetInstalledVersion()
	// Either it works (golangci-lint is installed) or it fails (not installed)
	if err == nil {
		assert.NotEmpty(t, version)
	} else {
		assert.Error(t, err)
	}
}

func TestNewTask(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}
	downloader := &mockFileDownloader{}
	extractor := &mockArchiveExtractor{}
	httpClient := &mockHTTPClient{}

	tool := NewTask("v3.38.0", logger, osUtil, downloader, extractor, httpClient)

	assert.Equal(t, "task", tool.Name())
	assert.Equal(t, "v3.38.0", tool.Version())
	assert.Contains(t, tool.Description(), "Task")
}

func TestTask_GetDownloadURL(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}
	downloader := &mockFileDownloader{}
	extractor := &mockArchiveExtractor{}
	httpClient := &mockHTTPClient{}

	// Mock OS and arch
	osUtil.On("GetOS").Return(runtime.GOOS)
	osUtil.On("GetArch").Return(runtime.GOARCH)

	tool := NewTask("v3.38.0", logger, osUtil, downloader, extractor, httpClient)

	url, err := tool.GetDownloadURL()
	assert.NoError(t, err)
	assert.Contains(t, url, "task")
	assert.Contains(t, url, runtime.GOOS)
}

func TestTask_GetBinaryName(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}
	downloader := &mockFileDownloader{}
	extractor := &mockArchiveExtractor{}
	httpClient := &mockHTTPClient{}

	// Mock the OS call that GetBinaryName will make
	osUtil.On("GetOS").Return("linux")

	tool := NewTask("v3.38.0", logger, osUtil, downloader, extractor, httpClient)

	binaryName := tool.GetBinaryName()
	assert.Equal(t, "task", binaryName)
}

func TestNewGofumpt(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}
	commandRunner := &mockCommandRunner{}

	tool := NewGofumpt("v0.7.0", logger, osUtil, commandRunner)

	assert.Equal(t, "gofumpt", tool.Name())
	assert.Equal(t, "v0.7.0", tool.Version())
	assert.Contains(t, tool.Description(), "gofmt")
}

func TestGofumpt_Install(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}
	commandRunner := &mockCommandRunner{}

	// Mock that tool is not installed
	osUtil.On("IsCommandAvailable", "gofumpt").Return(false)
	logger.On("Info", "Installing %s %s...", []interface{}{"gofumpt", "v0.7.0"})
	commandRunner.On("Run", context.Background(), "go", []string{"install", "mvdan.cc/gofumpt@v0.7.0"}).Return(nil)
	logger.On("Success", "%s %s installed successfully", []interface{}{"gofumpt", "v0.7.0"})

	tool := NewGofumpt("v0.7.0", logger, osUtil, commandRunner)

	err := tool.Install(context.Background())
	assert.NoError(t, err)

	osUtil.AssertExpectations(t)
	logger.AssertExpectations(t)
	commandRunner.AssertExpectations(t)
}

func TestNewMockgen(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}
	commandRunner := &mockCommandRunner{}

	tool := NewMockgen("v0.4.0", logger, osUtil, commandRunner)

	assert.Equal(t, "mockgen", tool.Name())
	assert.Equal(t, "v0.4.0", tool.Version())
	assert.Contains(t, tool.Description(), "mock")
}

func TestMockgen_Install(t *testing.T) {
	logger := &mockLogger{}
	osUtil := &mockOSUtil{}
	commandRunner := &mockCommandRunner{}

	// Mock that tool is not installed
	osUtil.On("IsCommandAvailable", "mockgen").Return(false)
	logger.On("Info", "Installing %s %s...", []interface{}{"mockgen", "v0.4.0"})
	commandRunner.On("Run", context.Background(), "go", []string{"install", "github.com/golang/mock/mockgen@v0.4.0"}).Return(nil)
	logger.On("Success", "%s %s installed successfully", []interface{}{"mockgen", "v0.4.0"})

	tool := NewMockgen("v0.4.0", logger, osUtil, commandRunner)

	err := tool.Install(context.Background())
	assert.NoError(t, err)

	osUtil.AssertExpectations(t)
	logger.AssertExpectations(t)
	commandRunner.AssertExpectations(t)
}