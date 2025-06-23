// Package tools provides Go-specific tool implementations
package tools

import (
	"context"
	"fmt"
	"os/exec"
	"strings"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
)

// GolangCILint represents the golangci-lint tool
type GolangCILint struct {
	*DownloadableTool
}

// NewGolangCILint creates a new golangci-lint tool installer
func NewGolangCILint(version string, logger interfaces.Logger, osUtil interfaces.OSUtil, downloader interfaces.FileDownloader, extractor interfaces.ArchiveExtractor, httpClient interfaces.HTTPClient) *GolangCILint {
	base := NewBaseTool(
		"golangci-lint",
		version,
		"Fast Go linters runner",
		"golangci-lint",
		logger,
		osUtil,
	)

	return &GolangCILint{
		DownloadableTool: NewDownloadableTool(base, downloader, extractor, httpClient),
	}
}

// GetDownloadURL returns the download URL for golangci-lint
func (g *GolangCILint) GetDownloadURL() (string, error) {
	os := g.osUtil.GetOS()
	arch := g.osUtil.GetArch()

	// Map OS names
	switch os {
	case "darwin":
		os = "darwin"
	case "linux":
		os = "linux"
	case "windows":
		os = "windows"
	default:
		return "", fmt.Errorf("unsupported OS: %s", os)
	}

	// Map arch names
	switch arch {
	case "amd64", "x86_64":
		arch = "amd64"
	case "arm64", "aarch64":
		arch = "arm64"
	default:
		return "", fmt.Errorf("unsupported architecture: %s", arch)
	}

	// Remove 'v' prefix if present
	version := strings.TrimPrefix(g.version, "v")

	url := fmt.Sprintf(
		"https://github.com/golangci/golangci-lint/releases/download/v%s/golangci-lint-%s-%s-%s.tar.gz",
		version, version, os, arch,
	)

	return url, nil
}

// GetBinaryName returns the binary name
func (g *GolangCILint) GetBinaryName() string {
	return "golangci-lint"
}

// GetInstalledVersion gets the installed version
func (g *GolangCILint) GetInstalledVersion() (string, error) {
	cmd := exec.Command(g.command, "--version")
	output, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("failed to get version: %w", err)
	}

	// Parse version from output like:
	// golangci-lint has version 1.55.2 built from ...
	lines := strings.Split(string(output), "\n")
	if len(lines) > 0 {
		parts := strings.Split(lines[0], " ")
		for i, part := range parts {
			if part == "version" && i+1 < len(parts) {
				return parts[i+1], nil
			}
		}
	}

	return "", fmt.Errorf("could not parse version from output")
}

// Task represents the Taskfile tool
type Task struct {
	*DownloadableTool
}

// NewTask creates a new Task tool installer
func NewTask(version string, logger interfaces.Logger, osUtil interfaces.OSUtil, downloader interfaces.FileDownloader, extractor interfaces.ArchiveExtractor, httpClient interfaces.HTTPClient) *Task {
	base := NewBaseTool(
		"task",
		version,
		"Task runner and build tool",
		"task",
		logger,
		osUtil,
	)

	return &Task{
		DownloadableTool: NewDownloadableTool(base, downloader, extractor, httpClient),
	}
}

// GetDownloadURL returns the download URL for Task
func (t *Task) GetDownloadURL() (string, error) {
	os := t.osUtil.GetOS()
	arch := t.osUtil.GetArch()

	// Map OS names
	switch os {
	case "darwin":
		os = "darwin"
	case "linux":
		os = "linux"
	case "windows":
		os = "windows"
	default:
		return "", fmt.Errorf("unsupported OS: %s", os)
	}

	// Map arch names
	switch arch {
	case "amd64", "x86_64":
		arch = "amd64"
	case "arm64", "aarch64":
		arch = "arm64"
	default:
		return "", fmt.Errorf("unsupported architecture: %s", arch)
	}

	// Remove 'v' prefix if present
	version := strings.TrimPrefix(t.version, "v")

	ext := "tar.gz"
	if os == "windows" {
		ext = "zip"
	}

	url := fmt.Sprintf(
		"https://github.com/go-task/task/releases/download/v%s/task_%s_%s_%s.%s",
		version, os, arch, version, ext,
	)

	return url, nil
}

// GetBinaryName returns the binary name
func (t *Task) GetBinaryName() string {
	if t.osUtil.GetOS() == "windows" {
		return "task.exe"
	}
	return "task"
}

// Gofumpt represents the gofumpt tool
type Gofumpt struct {
	*BaseTool
	commandRunner interfaces.CommandRunner
}

// NewGofumpt creates a new gofumpt tool installer
func NewGofumpt(version string, logger interfaces.Logger, osUtil interfaces.OSUtil, commandRunner interfaces.CommandRunner) *Gofumpt {
	return &Gofumpt{
		BaseTool: NewBaseTool(
			"gofumpt",
			version,
			"Stricter gofmt",
			"gofumpt",
			logger,
			osUtil,
		),
		commandRunner: commandRunner,
	}
}

// Install installs gofumpt using go install
func (g *Gofumpt) Install(ctx context.Context) error {
	if g.IsInstalled() {
		version, _ := g.GetInstalledVersion()
		g.logger.Info("%s is already installed: %s", g.name, version)
		return nil
	}

	g.logger.Info("Installing %s %s...", g.name, g.version)

	// Install using go install
	installCmd := fmt.Sprintf("mvdan.cc/gofumpt@%s", g.version)
	if err := g.commandRunner.Run(ctx, "go", "install", installCmd); err != nil {
		return fmt.Errorf("failed to install %s: %w", g.name, err)
	}

	g.logger.Success("%s %s installed successfully", g.name, g.version)
	return nil
}

// Mockgen represents the mockgen tool
type Mockgen struct {
	*BaseTool
	commandRunner interfaces.CommandRunner
}

// NewMockgen creates a new mockgen tool installer
func NewMockgen(version string, logger interfaces.Logger, osUtil interfaces.OSUtil, commandRunner interfaces.CommandRunner) *Mockgen {
	return &Mockgen{
		BaseTool: NewBaseTool(
			"mockgen",
			version,
			"GoMock mockgen tool",
			"mockgen",
			logger,
			osUtil,
		),
		commandRunner: commandRunner,
	}
}

// Install installs mockgen using go install
func (m *Mockgen) Install(ctx context.Context) error {
	if m.IsInstalled() {
		version, _ := m.GetInstalledVersion()
		m.logger.Info("%s is already installed: %s", m.name, version)
		return nil
	}

	m.logger.Info("Installing %s %s...", m.name, m.version)

	// Install using go install
	installCmd := fmt.Sprintf("github.com/golang/mock/mockgen@%s", m.version)
	if err := m.commandRunner.Run(ctx, "go", "install", installCmd); err != nil {
		return fmt.Errorf("failed to install %s: %w", m.name, err)
	}

	m.logger.Success("%s %s installed successfully", m.name, m.version)
	return nil
}
