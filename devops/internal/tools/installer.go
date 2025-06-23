// Package tools provides functionality for installing development tools
package tools

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strings"

	"github.com/raja-aiml/sematic-cache/devops/internal/httpclient"
	"github.com/raja-aiml/sematic-cache/devops/internal/logger"
)

// Tool represents a development tool to be installed
type Tool struct {
	Name        string
	Description string
	Version     string
	InstallFunc func(ctx context.Context, version string) error
}

// Installer handles tool installation
type Installer struct {
	logger      *logger.Logger
	httpClient  *httpclient.Client
	skipConfirm bool
}

// NewInstaller creates a new tool installer
func NewInstaller(skipConfirm bool) *Installer {
	return &Installer{
		logger:      logger.New(),
		httpClient:  httpclient.NewClient(),
		skipConfirm: skipConfirm,
	}
}

// InstallAll installs all missing development tools
func (i *Installer) InstallAll(ctx context.Context) error {
	tools := i.getTools()

	i.logger.Info("Development Tools Installer")
	i.logger.Info("==========================")

	// Detect OS
	osInfo := fmt.Sprintf("%s/%s", getOS(), getArch())
	i.logger.Info("Detected OS: %s", osInfo)

	// Check what's already installed
	i.logger.Info("Checking installed tools...")
	var missing []Tool

	for _, tool := range tools {
		if i.isInstalled(tool.Name) {
			version := i.getVersion(tool.Name)
			i.logger.Success("%s is already installed: %s", tool.Description, version)
		} else {
			missing = append(missing, tool)
		}
	}

	// Exit if all tools are installed
	if len(missing) == 0 {
		i.logger.Success("All tools are already installed!")
		return nil
	}

	// Show tools to install
	i.logger.Info("The following tools will be installed:")
	for _, tool := range missing {
		fmt.Printf("  - %s\n", tool.Description)
	}

	// Confirm installation
	if !i.skipConfirm {
		fmt.Print("Proceed with installation? [y/N] ")
		var response string
		fmt.Scanln(&response)
		if strings.ToLower(response) != "y" {
			return fmt.Errorf("installation cancelled")
		}
	}

	// Install missing tools
	for _, tool := range missing {
		if err := i.installTool(ctx, tool); err != nil {
			i.logger.Error("Failed to install %s: %v", tool.Name, err)
			return err
		}
	}

	i.logger.Success("Installation completed successfully!")

	// Verify installations
	i.logger.Info("Verifying installations...")
	for _, tool := range missing {
		if i.isInstalled(tool.Name) {
			version := i.getVersion(tool.Name)
			i.logger.Success("%s: %s", tool.Name, version)
		} else {
			i.logger.Error("%s installation failed", tool.Name)
		}
	}

	return nil
}

// getTools returns the list of tools to install
func (i *Installer) getTools() []Tool {
	return []Tool{
		{
			Name:        "task",
			Description: "Task (build automation)",
			Version:     "v3.31.0",
			InstallFunc: i.installTask,
		},
		{
			Name:        "golangci-lint",
			Description: "golangci-lint (Go linter)",
			Version:     "v1.55.2",
			InstallFunc: i.installGolangciLint,
		},
		{
			Name:        "k3d",
			Description: "k3d (local Kubernetes)",
			Version:     "v5.6.0",
			InstallFunc: i.installK3d,
		},
		{
			Name:        "helm",
			Description: "Helm (package manager)",
			Version:     "v3.13.2",
			InstallFunc: i.installHelm,
		},
		{
			Name:        "kustomize",
			Description: "Kustomize (config management)",
			Version:     "v5.3.0",
			InstallFunc: i.installKustomize,
		},
	}
}

// installTool installs a single tool
func (i *Installer) installTool(ctx context.Context, tool Tool) error {
	i.logger.Info("Installing %s %s...", tool.Name, tool.Version)
	if err := tool.InstallFunc(ctx, tool.Version); err != nil {
		return err
	}
	i.logger.Success("%s installed successfully", tool.Name)
	return nil
}

// installTask installs the Task build tool
func (i *Installer) installTask(ctx context.Context, version string) error {
	url := fmt.Sprintf("https://github.com/go-task/task/releases/download/%s/task_%s_%s.tar.gz",
		version, getOS(), getArch())

	return i.downloadAndExtractTarGz(ctx, url, "task", "/usr/local/bin/task")
}

// installGolangciLint installs golangci-lint
func (i *Installer) installGolangciLint(ctx context.Context, version string) error {
	// Use the official installer script
	cmd := exec.CommandContext(ctx, "sh", "-c",
		fmt.Sprintf("curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b /usr/local/bin %s", version))

	return cmd.Run()
}

// installK3d installs k3d
func (i *Installer) installK3d(ctx context.Context, version string) error {
	// Use the official installer script
	cmd := exec.CommandContext(ctx, "sh", "-c",
		fmt.Sprintf("curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | TAG=%s bash", version))

	return cmd.Run()
}

// installHelm installs Helm
func (i *Installer) installHelm(ctx context.Context, version string) error {
	url := fmt.Sprintf("https://get.helm.sh/helm-%s-%s-%s.tar.gz",
		version, getOS(), getArch())

	extractPath := fmt.Sprintf("%s-%s/helm", getOS(), getArch())
	return i.downloadAndExtractTarGz(ctx, url, extractPath, "/usr/local/bin/helm")
}

// installKustomize installs Kustomize
func (i *Installer) installKustomize(ctx context.Context, version string) error {
	url := fmt.Sprintf("https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%%2F%s/kustomize_%s_%s_%s.tar.gz",
		version, version, getOS(), getArch())

	return i.downloadAndExtractTarGz(ctx, url, "kustomize", "/usr/local/bin/kustomize")
}

// downloadAndExtractTarGz downloads and extracts a tar.gz file
func (i *Installer) downloadAndExtractTarGz(ctx context.Context, url, extractFile, destPath string) error {
	// Download file
	resp, err := i.httpClient.Get(ctx, url)
	if err != nil {
		return fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()

	// Create temporary file
	tmpFile, err := os.CreateTemp("", "tool-*.tar.gz")
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())

	// Save to temp file
	_, err = io.Copy(tmpFile, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to save file: %w", err)
	}
	tmpFile.Close()

	// Extract tar.gz
	file, err := os.Open(tmpFile.Name())
	if err != nil {
		return err
	}
	defer file.Close()

	gzr, err := gzip.NewReader(file)
	if err != nil {
		return err
	}
	defer gzr.Close()

	tr := tar.NewReader(gzr)

	// Find and extract the specific file
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		if strings.HasSuffix(header.Name, extractFile) {
			// Create temp file for extraction
			tmpExtract, err := os.CreateTemp("", "tool-*")
			if err != nil {
				return err
			}
			defer os.Remove(tmpExtract.Name())

			// Copy file contents
			_, err = io.Copy(tmpExtract, tr)
			if err != nil {
				return err
			}
			tmpExtract.Close()

			// Make executable
			if err := os.Chmod(tmpExtract.Name(), 0755); err != nil {
				return err
			}

			// Move to destination with sudo
			cmd := exec.CommandContext(ctx, "sudo", "mv", tmpExtract.Name(), destPath)
			return cmd.Run()
		}
	}

	return fmt.Errorf("file %s not found in archive", extractFile)
}

// isInstalled checks if a tool is installed
func (i *Installer) isInstalled(name string) bool {
	_, err := exec.LookPath(name)
	return err == nil
}

// getVersion gets the version of an installed tool
func (i *Installer) getVersion(name string) string {
	cmd := exec.Command(name, "version")
	output, err := cmd.Output()
	if err != nil {
		return "unknown"
	}

	lines := strings.Split(string(output), "\n")
	if len(lines) > 0 {
		return strings.TrimSpace(lines[0])
	}

	return "unknown"
}

// getOS returns the current OS in the format expected by tool downloads
func getOS() string {
	switch runtime.GOOS {
	case "darwin":
		return "darwin"
	case "linux":
		return "linux"
	case "windows":
		return "windows"
	default:
		return runtime.GOOS
	}
}

// getArch returns the current architecture in the format expected by tool downloads
func getArch() string {
	switch runtime.GOARCH {
	case "amd64":
		return "amd64"
	case "arm64":
		return "arm64"
	case "386":
		return "386"
	default:
		return runtime.GOARCH
	}
}
