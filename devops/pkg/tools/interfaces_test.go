// Package tools provides test interfaces
package tools

import (
	"context"
	"net/http"
	"time"
)

// Mock interfaces for testing (duplicated from internal/interfaces to avoid circular dependencies)

// Logger interface for logging operations
type Logger interface {
	Info(format string, args ...interface{})
	Success(format string, args ...interface{})
	Warning(format string, args ...interface{})
	Error(format string, args ...interface{})
	Debug(format string, args ...interface{})
}

// OSUtil interface for OS operations
type OSUtil interface {
	VerifyCommands(commands []string) ([]string, error)
	IsCommandAvailable(command string) bool
	GetOS() string
	GetArch() string
	GetHomeDir() string
	CreateTempDir(prefix string) (string, error)
	RemoveTempDir(path string) error
}

// FileDownloader interface for downloading files
type FileDownloader interface {
	Download(ctx context.Context, url, destPath string) error
	DownloadWithProgress(ctx context.Context, url, destPath string, progress chan<- float64) error
}

// ArchiveExtractor interface for extracting archives
type ArchiveExtractor interface {
	Extract(src, dest string) error
	ExtractFile(src, dest, filename string) error
}

// HTTPClient interface for HTTP operations
type HTTPClient interface {
	Get(ctx context.Context, url string) (*http.Response, error)
	Do(req *http.Request) (*http.Response, error)
	WaitForHTTP(ctx context.Context, url string, timeout time.Duration) error
	WaitForPort(ctx context.Context, host string, port int, timeout time.Duration) error
	CheckHealth(ctx context.Context, url string, expectedStatus int) error
}

// CommandRunner interface for running external commands
type CommandRunner interface {
	Run(ctx context.Context, name string, args ...string) error
	RunWithOutput(ctx context.Context, name string, args ...string) (string, error)
	RunWithEnv(ctx context.Context, env []string, name string, args ...string) error
}

// ToolInstaller interface for tool installation
type ToolInstaller interface {
	Name() string
	Version() string
	Description() string
	IsInstalled() bool
	GetInstalledVersion() (string, error)
	Install(ctx context.Context) error
	Uninstall(ctx context.Context) error
	Validate() error
}
