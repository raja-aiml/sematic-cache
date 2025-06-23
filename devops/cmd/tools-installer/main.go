// tools-installer installs required development tools
package main

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/raja-aiml/sematic-cache/devops/pkg/devops/logger"
	"github.com/raja-aiml/sematic-cache/devops/pkg/devops/osutil"
)

// Tool represents a development tool to install
type Tool struct {
	Name        string
	Description string
	Version     string
	Installer   func(ctx context.Context) error
}

// ToolInstaller manages tool installations
type ToolInstaller struct {
	logger     *logger.Logger
	httpClient *http.Client
	tools      []Tool
}

// NewToolInstaller creates a new tool installer
func NewToolInstaller() *ToolInstaller {
	ti := &ToolInstaller{
		logger: logger.New(),
		httpClient: &http.Client{
			Timeout: 5 * time.Minute,
		},
	}

	// Define tools with their installers
	ti.tools = []Tool{
		{
			Name:        "task",
			Description: "Task (build automation)",
			Version:     "v3.31.0",
			Installer:   ti.installTask,
		},
		{
			Name:        "golangci-lint",
			Description: "golangci-lint (Go linter)",
			Version:     "v1.55.2",
			Installer:   ti.installGolangciLint,
		},
		{
			Name:        "k3d",
			Description: "k3d (local Kubernetes)",
			Version:     "v5.6.0",
			Installer:   ti.installK3d,
		},
		{
			Name:        "helm",
			Description: "Helm (package manager)",
			Version:     "v3.13.2",
			Installer:   ti.installHelm,
		},
		{
			Name:        "kustomize",
			Description: "Kustomize (config management)",
			Version:     "v5.3.0",
			Installer:   ti.installKustomize,
		},
	}

	return ti
}

// Run executes the tool installation process
func (ti *ToolInstaller) Run(ctx context.Context, skipConfirmation bool) error {
	ti.logger.Info("Development Tools Installer")
	ti.logger.Info("==========================")

	// Check for required tools
	required := []string{"curl", "tar"}
	if runtime.GOOS != "windows" {
		required = append(required, "sudo")
	}

	if missing, err := osutil.VerifyCommands(required); err != nil {
		ti.logger.Error("Missing required tools: %v", missing)
		return err
	}

	// Detect platform
	platform := osutil.GetPlatform()
	ti.logger.Info("Detected platform: %s", platform.String())

	// Check what's already installed
	ti.logger.Info("Checking installed tools...")
	var toInstall []Tool

	for _, tool := range ti.tools {
		if osutil.CommandExists(tool.Name) {
			version := ti.getToolVersion(tool.Name)
			ti.logger.Success("%s is already installed: %s", tool.Description, version)
		} else {
			toInstall = append(toInstall, tool)
		}
	}

	// Exit if all tools are installed
	if len(toInstall) == 0 {
		ti.logger.Success("All tools are already installed!")
		return nil
	}

	// Show tools to install
	ti.logger.Info("The following tools will be installed:")
	for _, tool := range toInstall {
		fmt.Printf("  - %s (version: %s)\n", tool.Description, tool.Version)
	}

	// Confirm installation
	if !skipConfirmation {
		if !ti.confirm("Proceed with installation?") {
			ti.logger.Info("Installation cancelled")
			return nil
		}
	}

	// Install missing tools
	for _, tool := range toInstall {
		start := time.Now()
		ti.logger.Info("Installing %s...", tool.Description)
		
		if err := tool.Installer(ctx); err != nil {
			ti.logger.Error("Failed to install %s: %v", tool.Name, err)
			return err
		}
		
		ti.logger.Success("%s installed successfully (took %v)", tool.Description, time.Since(start))
	}

	ti.logger.Success("Installation completed successfully!")

	// Verify installations
	ti.logger.Info("Verifying installations...")
	for _, tool := range toInstall {
		if osutil.CommandExists(tool.Name) {
			version := ti.getToolVersion(tool.Name)
			ti.logger.Success("%s: %s", tool.Name, version)
		} else {
			ti.logger.Error("%s installation verification failed", tool.Name)
		}
	}

	return nil
}

// getToolVersion gets the version of an installed tool
func (ti *ToolInstaller) getToolVersion(name string) string {
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

// confirm asks for user confirmation
func (ti *ToolInstaller) confirm(prompt string) bool {
	fmt.Printf("%s [y/N]: ", prompt)
	var response string
	fmt.Scanln(&response)
	return strings.ToLower(response) == "y" || strings.ToLower(response) == "yes"
}

// downloadAndExtract downloads a tar.gz file and extracts it
func (ti *ToolInstaller) downloadAndExtract(ctx context.Context, url string, destDir string) error {
	ti.logger.Debug("Downloading from: %s", url)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := ti.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed with status: %d", resp.StatusCode)
	}

	// Create gzip reader
	gzReader, err := gzip.NewReader(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer gzReader.Close()

	// Create tar reader
	tarReader := tar.NewReader(gzReader)

	// Extract files
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to read tar: %w", err)
		}

		target := filepath.Join(destDir, header.Name)

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0755); err != nil {
				return fmt.Errorf("failed to create directory: %w", err)
			}
		case tar.TypeReg:
			file, err := os.OpenFile(target, os.O_CREATE|os.O_RDWR, os.FileMode(header.Mode))
			if err != nil {
				return fmt.Errorf("failed to create file: %w", err)
			}
			
			if _, err := io.Copy(file, tarReader); err != nil {
				file.Close()
				return fmt.Errorf("failed to write file: %w", err)
			}
			file.Close()
		}
	}

	return nil
}

// moveToPath moves a file to /usr/local/bin with sudo if needed
func (ti *ToolInstaller) moveToPath(source, name string) error {
	dest := filepath.Join("/usr/local/bin", name)
	
	// Try direct move first
	if err := os.Rename(source, dest); err == nil {
		return nil
	}

	// Fall back to sudo mv
	if runtime.GOOS != "windows" {
		cmd := exec.Command("sudo", "mv", source, dest)
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("failed to move file to PATH: %w", err)
		}
	}

	return nil
}

// Tool-specific installers

func (ti *ToolInstaller) installTask(ctx context.Context) error {
	os := osutil.GetOS()
	arch := osutil.GetArch()
	version := ti.getToolByName("task").Version
	
	url := fmt.Sprintf("https://github.com/go-task/task/releases/download/%s/task_%s_%s.tar.gz",
		version, os, arch)

	tmpDir, err := os.MkdirTemp("", "task-install")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)

	if err := ti.downloadAndExtract(ctx, url, tmpDir); err != nil {
		return err
	}

	return ti.moveToPath(filepath.Join(tmpDir, "task"), "task")
}

func (ti *ToolInstaller) installGolangciLint(ctx context.Context) error {
	version := ti.getToolByName("golangci-lint").Version
	
	// Use the official install script
	scriptURL := "https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh"
	
	req, err := http.NewRequestWithContext(ctx, "GET", scriptURL, nil)
	if err != nil {
		return err
	}

	resp, err := ti.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	script, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	// Save and execute the script
	tmpFile, err := os.CreateTemp("", "golangci-lint-install.sh")
	if err != nil {
		return err
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.Write(script); err != nil {
		return err
	}
	tmpFile.Close()

	cmd := exec.Command("sh", tmpFile.Name(), "-b", "/usr/local/bin", version)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func (ti *ToolInstaller) installK3d(ctx context.Context) error {
	version := ti.getToolByName("k3d").Version
	
	// Use the official install script
	scriptURL := "https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh"
	
	cmd := exec.Command("sh", "-c",
		fmt.Sprintf("curl -s %s | TAG=%s bash", scriptURL, version))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(), fmt.Sprintf("TAG=%s", version))
	
	return cmd.Run()
}

func (ti *ToolInstaller) installHelm(ctx context.Context) error {
	os := osutil.GetOS()
	arch := osutil.GetArch()
	version := ti.getToolByName("helm").Version
	
	url := fmt.Sprintf("https://get.helm.sh/helm-%s-%s-%s.tar.gz",
		version, os, arch)

	tmpDir, err := os.MkdirTemp("", "helm-install")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)

	if err := ti.downloadAndExtract(ctx, url, tmpDir); err != nil {
		return err
	}

	helmPath := filepath.Join(tmpDir, fmt.Sprintf("%s-%s", os, arch), "helm")
	return ti.moveToPath(helmPath, "helm")
}

func (ti *ToolInstaller) installKustomize(ctx context.Context) error {
	os := osutil.GetOS()
	arch := osutil.GetArch()
	version := ti.getToolByName("kustomize").Version
	
	// Kustomize URL encoding is special
	versionEncoded := strings.ReplaceAll(version, "/", "%2F")
	url := fmt.Sprintf("https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%s/kustomize_%s_%s_%s.tar.gz",
		versionEncoded, version, os, arch)

	tmpDir, err := os.MkdirTemp("", "kustomize-install")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)

	if err := ti.downloadAndExtract(ctx, url, tmpDir); err != nil {
		return err
	}

	return ti.moveToPath(filepath.Join(tmpDir, "kustomize"), "kustomize")
}

// getToolByName returns a tool by name
func (ti *ToolInstaller) getToolByName(name string) *Tool {
	for i := range ti.tools {
		if ti.tools[i].Name == name {
			return &ti.tools[i]
		}
	}
	return nil
}

func main() {
	var (
		skipConfirmation = flag.Bool("skip-confirmation", false, "Skip installation confirmation")
		debugMode        = flag.Bool("debug", false, "Enable debug logging")
	)
	flag.Parse()

	installer := NewToolInstaller()
	
	if *debugMode {
		installer.logger.SetLevel(logger.DebugLevel)
	}

	ctx := context.Background()
	if err := installer.Run(ctx, *skipConfirmation); err != nil {
		installer.logger.Fatal("Installation failed: %v", err)
	}
}