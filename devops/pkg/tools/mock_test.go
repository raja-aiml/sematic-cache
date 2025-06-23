// Package tools provides mock implementations for testing
package tools

import (
	"context"
	"net/http"
	"time"

	"github.com/stretchr/testify/mock"
)

// mockLogger implements the Logger interface for testing
type mockLogger struct {
	mock.Mock
}

func (m *mockLogger) Info(format string, args ...interface{}) {
	m.Called(format, args)
}

func (m *mockLogger) Success(format string, args ...interface{}) {
	m.Called(format, args)
}

func (m *mockLogger) Warning(format string, args ...interface{}) {
	m.Called(format, args)
}

func (m *mockLogger) Error(format string, args ...interface{}) {
	m.Called(format, args)
}

func (m *mockLogger) Debug(format string, args ...interface{}) {
	m.Called(format, args)
}

// mockOSUtil implements the OSUtil interface for testing
type mockOSUtil struct {
	mock.Mock
}

func (m *mockOSUtil) VerifyCommands(commands []string) ([]string, error) {
	args := m.Called(commands)
	return args.Get(0).([]string), args.Error(1)
}

func (m *mockOSUtil) IsCommandAvailable(command string) bool {
	args := m.Called(command)
	return args.Bool(0)
}

func (m *mockOSUtil) GetOS() string {
	args := m.Called()
	return args.String(0)
}

func (m *mockOSUtil) GetArch() string {
	args := m.Called()
	return args.String(0)
}

func (m *mockOSUtil) GetHomeDir() string {
	args := m.Called()
	return args.String(0)
}

func (m *mockOSUtil) CreateTempDir(prefix string) (string, error) {
	args := m.Called(prefix)
	return args.String(0), args.Error(1)
}

func (m *mockOSUtil) RemoveTempDir(path string) error {
	args := m.Called(path)
	return args.Error(0)
}

// mockFileDownloader implements the FileDownloader interface for testing
type mockFileDownloader struct {
	mock.Mock
}

func (m *mockFileDownloader) Download(ctx context.Context, url, destPath string) error {
	args := m.Called(ctx, url, destPath)
	return args.Error(0)
}

func (m *mockFileDownloader) DownloadWithProgress(ctx context.Context, url, destPath string, progress chan<- float64) error {
	args := m.Called(ctx, url, destPath, progress)
	return args.Error(0)
}

// mockArchiveExtractor implements the ArchiveExtractor interface for testing
type mockArchiveExtractor struct {
	mock.Mock
}

func (m *mockArchiveExtractor) Extract(src, dest string) error {
	args := m.Called(src, dest)
	return args.Error(0)
}

func (m *mockArchiveExtractor) ExtractFile(src, dest, filename string) error {
	args := m.Called(src, dest, filename)
	return args.Error(0)
}

// mockHTTPClient implements the HTTPClient interface for testing
type mockHTTPClient struct {
	mock.Mock
}

func (m *mockHTTPClient) Get(ctx context.Context, url string) (*http.Response, error) {
	args := m.Called(ctx, url)
	if resp := args.Get(0); resp != nil {
		return resp.(*http.Response), args.Error(1)
	}
	return nil, args.Error(1)
}

func (m *mockHTTPClient) Do(req *http.Request) (*http.Response, error) {
	args := m.Called(req)
	if resp := args.Get(0); resp != nil {
		return resp.(*http.Response), args.Error(1)
	}
	return nil, args.Error(1)
}

func (m *mockHTTPClient) WaitForHTTP(ctx context.Context, url string, timeout time.Duration) error {
	args := m.Called(ctx, url, timeout)
	return args.Error(0)
}

func (m *mockHTTPClient) WaitForPort(ctx context.Context, host string, port int, timeout time.Duration) error {
	args := m.Called(ctx, host, port, timeout)
	return args.Error(0)
}

func (m *mockHTTPClient) CheckHealth(ctx context.Context, url string, expectedStatus int) error {
	args := m.Called(ctx, url, expectedStatus)
	return args.Error(0)
}

// mockCommandRunner implements the CommandRunner interface for testing
type mockCommandRunner struct {
	mock.Mock
}

func (m *mockCommandRunner) Run(ctx context.Context, name string, args ...string) error {
	mArgs := m.Called(ctx, name, args)
	return mArgs.Error(0)
}

func (m *mockCommandRunner) RunWithOutput(ctx context.Context, name string, args ...string) (string, error) {
	mArgs := m.Called(ctx, name, args)
	return mArgs.String(0), mArgs.Error(1)
}

func (m *mockCommandRunner) RunWithEnv(ctx context.Context, env []string, name string, args ...string) error {
	mArgs := m.Called(ctx, env, name, args)
	return mArgs.Error(0)
}

// mockToolInstaller implements the ToolInstaller interface for testing
type mockToolInstaller struct {
	mock.Mock
}

func (m *mockToolInstaller) Name() string {
	args := m.Called()
	return args.String(0)
}

func (m *mockToolInstaller) Version() string {
	args := m.Called()
	return args.String(0)
}

func (m *mockToolInstaller) Description() string {
	args := m.Called()
	return args.String(0)
}

func (m *mockToolInstaller) IsInstalled() bool {
	args := m.Called()
	return args.Bool(0)
}

func (m *mockToolInstaller) GetInstalledVersion() (string, error) {
	args := m.Called()
	return args.String(0), args.Error(1)
}

func (m *mockToolInstaller) Install(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

func (m *mockToolInstaller) Uninstall(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

func (m *mockToolInstaller) Validate() error {
	args := m.Called()
	return args.Error(0)
}
