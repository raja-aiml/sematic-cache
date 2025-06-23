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

// K3D represents the k3d tool
type K3D struct {
	*BaseTool
	commandRunner interfaces.CommandRunner
}

// NewK3D creates a new k3d tool installer
func NewK3D(version string, logger interfaces.Logger, osUtil interfaces.OSUtil, commandRunner interfaces.CommandRunner) *K3D {
	return &K3D{
		BaseTool: NewBaseTool(
			"k3d",
			version,
			"k3d - Little helper to run Rancher Lab's k3s in Docker",
			"k3d",
			logger,
			osUtil,
		),
		commandRunner: commandRunner,
	}
}

// Install installs k3d using curl script
func (k *K3D) Install(ctx context.Context) error {
	if k.IsInstalled() {
		version, _ := k.GetInstalledVersion()
		k.logger.Info("%s is already installed: %s", k.name, version)
		return nil
	}

	k.logger.Info("Installing %s %s...", k.name, k.version)

	// Install using official script
	installCmd := fmt.Sprintf("curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | TAG=%s bash", k.version)
	if err := k.commandRunner.Run(ctx, "sh", "-c", installCmd); err != nil {
		return fmt.Errorf("failed to install %s: %w", k.name, err)
	}

	k.logger.Success("%s %s installed successfully", k.name, k.version)
	return nil
}

// Helm represents the helm tool
type Helm struct {
	*DownloadableTool
}

// NewHelm creates a new helm tool installer
func NewHelm(version string, logger interfaces.Logger, osUtil interfaces.OSUtil, downloader interfaces.FileDownloader, extractor interfaces.ArchiveExtractor, httpClient interfaces.HTTPClient) *Helm {
	base := NewBaseTool(
		"helm",
		version,
		"The Kubernetes Package Manager",
		"helm",
		logger,
		osUtil,
	)

	return &Helm{
		DownloadableTool: NewDownloadableTool(base, downloader, extractor, httpClient),
	}
}

// GetDownloadURL returns the download URL for Helm
func (h *Helm) GetDownloadURL() (string, error) {
	os := h.osUtil.GetOS()
	arch := h.osUtil.GetArch()

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
	version := strings.TrimPrefix(h.version, "v")

	url := fmt.Sprintf(
		"https://get.helm.sh/helm-v%s-%s-%s.tar.gz",
		version, os, arch,
	)

	return url, nil
}

// GetBinaryName returns the binary name
func (h *Helm) GetBinaryName() string {
	if h.osUtil.GetOS() == "windows" {
		return "helm.exe"
	}
	return "helm"
}

// Kustomize represents the kustomize tool
type Kustomize struct {
	*DownloadableTool
}

// NewKustomize creates a new kustomize tool installer
func NewKustomize(version string, logger interfaces.Logger, osUtil interfaces.OSUtil, downloader interfaces.FileDownloader, extractor interfaces.ArchiveExtractor, httpClient interfaces.HTTPClient) *Kustomize {
	base := NewBaseTool(
		"kustomize",
		version,
		"Kubernetes configuration management tool",
		"kustomize",
		logger,
		osUtil,
	)

	return &Kustomize{
		DownloadableTool: NewDownloadableTool(base, downloader, extractor, httpClient),
	}
}

// GetDownloadURL returns the download URL for Kustomize
func (k *Kustomize) GetDownloadURL() (string, error) {
	os := k.osUtil.GetOS()
	arch := k.osUtil.GetArch()

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
	version := strings.TrimPrefix(k.version, "v")

	url := fmt.Sprintf(
		"https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%%2Fv%s/kustomize_v%s_%s_%s.tar.gz",
		version, version, os, arch,
	)

	return url, nil
}

// GetBinaryName returns the binary name
func (k *Kustomize) GetBinaryName() string {
	if k.osUtil.GetOS() == "windows" {
		return "kustomize.exe"
	}
	return "kustomize"
}
