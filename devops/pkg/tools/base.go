// Package tools provides generic tool installation functionality
package tools

import (
	"context"
	"fmt"
	"os/exec"
	"strings"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
)

// BaseTool provides common functionality for all tools
type BaseTool struct {
	name        string
	version     string
	description string
	command     string
	logger      interfaces.Logger
	osUtil      interfaces.OSUtil
}

// NewBaseTool creates a new base tool
func NewBaseTool(name, version, description, command string, logger interfaces.Logger, osUtil interfaces.OSUtil) *BaseTool {
	return &BaseTool{
		name:        name,
		version:     version,
		description: description,
		command:     command,
		logger:      logger,
		osUtil:      osUtil,
	}
}

// Name returns the tool name
func (t *BaseTool) Name() string {
	return t.name
}

// Version returns the tool version
func (t *BaseTool) Version() string {
	return t.version
}

// Description returns the tool description
func (t *BaseTool) Description() string {
	return t.description
}

// Logger returns the logger instance
func (t *BaseTool) Logger() interfaces.Logger {
	return t.logger
}

// IsInstalled checks if the tool is installed
func (t *BaseTool) IsInstalled() bool {
	return t.osUtil.IsCommandAvailable(t.command)
}

// GetInstalledVersion gets the installed version of the tool
func (t *BaseTool) GetInstalledVersion() (string, error) {
	cmd := exec.Command(t.command, "--version")
	output, err := cmd.Output()
	if err != nil {
		// Try with -version flag
		cmd = exec.Command(t.command, "-version")
		output, err = cmd.Output()
		if err != nil {
			return "", fmt.Errorf("failed to get version: %w", err)
		}
	}

	// Extract version from output (this is a simple implementation)
	// Specific tools can override this method for custom parsing
	lines := strings.Split(string(output), "\n")
	if len(lines) > 0 {
		return strings.TrimSpace(lines[0]), nil
	}

	return string(output), nil
}

// Validate checks if the tool is properly installed
func (t *BaseTool) Validate() error {
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

// Uninstall is not implemented in base tool
func (t *BaseTool) Uninstall(ctx context.Context) error {
	return fmt.Errorf("uninstall not implemented for %s", t.name)
}

// DownloadableTool extends BaseTool with download capabilities
type DownloadableTool struct {
	*BaseTool
	downloader interfaces.FileDownloader
	extractor  interfaces.ArchiveExtractor
	httpClient interfaces.HTTPClient
}

// NewDownloadableTool creates a new downloadable tool
func NewDownloadableTool(base *BaseTool, downloader interfaces.FileDownloader, extractor interfaces.ArchiveExtractor, httpClient interfaces.HTTPClient) *DownloadableTool {
	return &DownloadableTool{
		BaseTool:   base,
		downloader: downloader,
		extractor:  extractor,
		httpClient: httpClient,
	}
}

// GetDownloadURL returns the download URL for the tool
// This should be overridden by specific tool implementations
func (t *DownloadableTool) GetDownloadURL() (string, error) {
	return "", fmt.Errorf("GetDownloadURL must be implemented by specific tool")
}

// GetBinaryName returns the binary name inside the archive
// This should be overridden by specific tool implementations
func (t *DownloadableTool) GetBinaryName() string {
	return t.command
}

// Install downloads and installs the tool
func (t *DownloadableTool) Install(ctx context.Context) error {
	if t.IsInstalled() {
		version, _ := t.GetInstalledVersion()
		t.logger.Info("%s is already installed: %s", t.name, version)
		return nil
	}

	url, err := t.GetDownloadURL()
	if err != nil {
		return fmt.Errorf("failed to get download URL: %w", err)
	}

	t.logger.Info("Downloading %s %s...", t.name, t.version)

	// Create temp directory
	tempDir, err := t.osUtil.CreateTempDir(t.name)
	if err != nil {
		return fmt.Errorf("failed to create temp dir: %w", err)
	}
	defer t.osUtil.RemoveTempDir(tempDir)

	// Download file
	archivePath := fmt.Sprintf("%s/%s-archive", tempDir, t.name)
	if err := t.downloader.Download(ctx, url, archivePath); err != nil {
		return fmt.Errorf("failed to download %s: %w", t.name, err)
	}

	// Extract if needed
	if strings.HasSuffix(url, ".tar.gz") || strings.HasSuffix(url, ".tgz") {
		if err := t.extractor.Extract(archivePath, tempDir); err != nil {
			return fmt.Errorf("failed to extract %s: %w", t.name, err)
		}
	}

	// Install binary
	if err := t.installBinary(tempDir); err != nil {
		return fmt.Errorf("failed to install %s: %w", t.name, err)
	}

	t.logger.Success("%s %s installed successfully", t.name, t.version)
	return nil
}

// installBinary installs the binary to the system
func (t *DownloadableTool) installBinary(tempDir string) error {
	// This is a generic implementation
	// Specific tools can override this for custom installation
	binaryName := t.GetBinaryName()
	sourcePath := fmt.Sprintf("%s/%s", tempDir, binaryName)

	// Check if binary exists directly
	if !fileExists(sourcePath) {
		// Try to find it in subdirectories
		// This is simplified - real implementation would be more robust
		sourcePath = fmt.Sprintf("%s/%s-%s-%s/%s", tempDir, t.name, t.osUtil.GetOS(), t.osUtil.GetArch(), binaryName)
	}

	if !fileExists(sourcePath) {
		return fmt.Errorf("binary not found: %s", binaryName)
	}

	// Make executable
	if err := makeExecutable(sourcePath); err != nil {
		return fmt.Errorf("failed to make executable: %w", err)
	}

	// Move to /usr/local/bin (this should be configurable)
	destPath := fmt.Sprintf("/usr/local/bin/%s", binaryName)
	if err := moveFile(sourcePath, destPath); err != nil {
		return fmt.Errorf("failed to move binary: %w", err)
	}

	return nil
}

// Helper functions

func fileExists(path string) bool {
	_, err := exec.Command("test", "-f", path).Output()
	return err == nil
}

func makeExecutable(path string) error {
	return exec.Command("chmod", "+x", path).Run()
}

func moveFile(src, dst string) error {
	// Try to use sudo if needed
	err := exec.Command("mv", src, dst).Run()
	if err != nil {
		return exec.Command("sudo", "mv", src, dst).Run()
	}
	return nil
}
