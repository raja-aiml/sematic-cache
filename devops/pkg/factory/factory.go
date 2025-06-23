// Package factory provides factory functions for creating instances with dependency injection
package factory

import (
	"context"
	"fmt"
	"time"

	"github.com/spf13/viper"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
	"github.com/raja-aiml/sematic-cache/devops/pkg/command"
	"github.com/raja-aiml/sematic-cache/devops/pkg/docker"
	"github.com/raja-aiml/sematic-cache/devops/pkg/downloader"
	"github.com/raja-aiml/sematic-cache/devops/pkg/extractor"
	"github.com/raja-aiml/sematic-cache/devops/pkg/httpclient"
	"github.com/raja-aiml/sematic-cache/devops/pkg/kubernetes"
	"github.com/raja-aiml/sematic-cache/devops/pkg/logger"
	"github.com/raja-aiml/sematic-cache/devops/pkg/osutil"
	"github.com/raja-aiml/sematic-cache/devops/pkg/tools"
)

// Config holds configuration for creating instances
type Config struct {
	// Logger configuration
	LogLevel string
	LogColor bool

	// HTTP client configuration
	HTTPTimeout time.Duration
	HTTPRetries int

	// Tool versions
	ToolVersions map[string]string

	// Kubernetes configuration
	KubeConfig string

	// Docker configuration
	DockerHost string
}

// DefaultConfig returns default configuration
func DefaultConfig() *Config {
	return &Config{
		LogLevel:    "info",
		LogColor:    true,
		HTTPTimeout: 5 * time.Minute,
		HTTPRetries: 3,
		ToolVersions: map[string]string{
			"task":          "v3.31.0",
			"golangci-lint": "v1.55.2",
			"gofumpt":       "v0.5.0",
			"mockgen":       "v1.6.0",
			"k3d":           "v5.6.0",
			"helm":          "v3.13.2",
			"kustomize":     "v5.3.0",
		},
	}
}

// Factory creates instances with proper dependency injection
type Factory struct {
	config        *Config
	logger        interfaces.Logger
	httpClient    interfaces.HTTPClient
	dockerClient  interfaces.DockerClient
	k8sClient     interfaces.KubernetesClient
	osUtil        interfaces.OSUtil
	downloader    interfaces.FileDownloader
	extractor     interfaces.ArchiveExtractor
	commandRunner interfaces.CommandRunner
}

// NewFactory creates a new factory instance
func NewFactory(config *Config) (*Factory, error) {
	if config == nil {
		config = DefaultConfig()
	}

	f := &Factory{
		config: config,
	}

	// Initialize core components
	if err := f.initializeComponents(); err != nil {
		return nil, fmt.Errorf("failed to initialize factory: %w", err)
	}

	return f, nil
}

// initializeComponents initializes all core components
func (f *Factory) initializeComponents() error {
	// Create logger first as other components depend on it
	f.logger = f.createLogger()

	// Create other core components
	f.osUtil = f.createOSUtil()
	f.httpClient = f.createHTTPClient()
	f.downloader = f.createFileDownloader()
	f.extractor = f.createArchiveExtractor()
	f.commandRunner = f.createCommandRunner()

	return nil
}

// GetLogger returns the logger instance
func (f *Factory) GetLogger() interfaces.Logger {
	return f.logger
}

// GetHTTPClient returns the HTTP client instance
func (f *Factory) GetHTTPClient() interfaces.HTTPClient {
	return f.httpClient
}

// GetOSUtil returns the OS utility instance
func (f *Factory) GetOSUtil() interfaces.OSUtil {
	return f.osUtil
}

// GetDockerClient returns the Docker client instance
func (f *Factory) GetDockerClient() (interfaces.DockerClient, error) {
	if f.dockerClient == nil {
		client, err := docker.NewClient(f.logger)
		if err != nil {
			return nil, err
		}
		f.dockerClient = client
	}
	return f.dockerClient, nil
}

// GetKubernetesClient returns the Kubernetes client instance
func (f *Factory) GetKubernetesClient() (interfaces.KubernetesClient, error) {
	if f.k8sClient == nil {
		client, err := kubernetes.NewClient(f.logger, f.config.KubeConfig)
		if err != nil {
			return nil, err
		}
		f.k8sClient = client
	}
	return f.k8sClient, nil
}

// CreateToolRegistry creates a new tool registry with all tools
func (f *Factory) CreateToolRegistry() (interfaces.ToolRegistry, error) {
	registry := tools.NewRegistry(f.logger)

	// Register all tools
	toolsToRegister := []interfaces.ToolInstaller{
		f.createTaskTool(),
		f.createGolangCILintTool(),
		f.createGofumptTool(),
		f.createMockgenTool(),
		f.createK3DTool(),
		f.createHelmTool(),
		f.createKustomizeTool(),
	}

	for _, tool := range toolsToRegister {
		if err := registry.Register(tool); err != nil {
			return nil, fmt.Errorf("failed to register %s: %w", tool.Name(), err)
		}
	}

	return registry, nil
}

// CreateCustomToolRegistry creates a tool registry with custom tools
func (f *Factory) CreateCustomToolRegistry(toolConfigs []ToolConfig) (interfaces.ToolRegistry, error) {
	registry := tools.NewRegistry(f.logger)

	for _, config := range toolConfigs {
		tool, err := f.createCustomTool(config)
		if err != nil {
			return nil, fmt.Errorf("failed to create tool %s: %w", config.Name, err)
		}

		if err := registry.Register(tool); err != nil {
			return nil, fmt.Errorf("failed to register %s: %w", config.Name, err)
		}
	}

	return registry, nil
}

// ToolConfig defines configuration for a custom tool
type ToolConfig struct {
	Name        string
	Version     string
	Description string
	Command     string
	Type        string // "downloadable", "go-install", "custom"
	URL         string // For downloadable tools
	Package     string // For go-install tools
}

// Tool creation methods

func (f *Factory) createTaskTool() interfaces.ToolInstaller {
	version := f.config.ToolVersions["task"]
	return tools.NewTask(version, f.logger, f.osUtil, f.downloader, f.extractor, f.httpClient)
}

func (f *Factory) createGolangCILintTool() interfaces.ToolInstaller {
	version := f.config.ToolVersions["golangci-lint"]
	return tools.NewGolangCILint(version, f.logger, f.osUtil, f.downloader, f.extractor, f.httpClient)
}

func (f *Factory) createGofumptTool() interfaces.ToolInstaller {
	version := f.config.ToolVersions["gofumpt"]
	return tools.NewGofumpt(version, f.logger, f.osUtil, f.commandRunner)
}

func (f *Factory) createMockgenTool() interfaces.ToolInstaller {
	version := f.config.ToolVersions["mockgen"]
	return tools.NewMockgen(version, f.logger, f.osUtil, f.commandRunner)
}

func (f *Factory) createK3DTool() interfaces.ToolInstaller {
	version := f.config.ToolVersions["k3d"]
	return tools.NewK3D(version, f.logger, f.osUtil, f.commandRunner)
}

func (f *Factory) createHelmTool() interfaces.ToolInstaller {
	version := f.config.ToolVersions["helm"]
	return tools.NewHelm(version, f.logger, f.osUtil, f.downloader, f.extractor, f.httpClient)
}

func (f *Factory) createKustomizeTool() interfaces.ToolInstaller {
	version := f.config.ToolVersions["kustomize"]
	return tools.NewKustomize(version, f.logger, f.osUtil, f.downloader, f.extractor, f.httpClient)
}

func (f *Factory) createCustomTool(config ToolConfig) (interfaces.ToolInstaller, error) {
	switch config.Type {
	case "downloadable":
		return f.createDownloadableTool(config), nil
	case "go-install":
		return f.createGoInstallTool(config), nil
	default:
		return nil, fmt.Errorf("unsupported tool type: %s", config.Type)
	}
}

func (f *Factory) createDownloadableTool(config ToolConfig) interfaces.ToolInstaller {
	base := tools.NewBaseTool(
		config.Name,
		config.Version,
		config.Description,
		config.Command,
		f.logger,
		f.osUtil,
	)

	// Create a custom downloadable tool
	return &customDownloadableTool{
		DownloadableTool: tools.NewDownloadableTool(base, f.downloader, f.extractor, f.httpClient),
		url:              config.URL,
	}
}

func (f *Factory) createGoInstallTool(config ToolConfig) interfaces.ToolInstaller {
	return &goInstallTool{
		BaseTool:      tools.NewBaseTool(config.Name, config.Version, config.Description, config.Command, f.logger, f.osUtil),
		packageName:   config.Package,
		commandRunner: f.commandRunner,
	}
}

// Component creation methods

func (f *Factory) createLogger() interfaces.Logger {
	useColor := !viper.GetBool("log.noColor")
	level := logger.ParseLevel(f.config.LogLevel)
	return logger.NewWithOptions(level, useColor)
}

func (f *Factory) createOSUtil() interfaces.OSUtil {
	return osutil.New()
}

func (f *Factory) createHTTPClient() interfaces.HTTPClient {
	return httpclient.NewWithOptions(f.logger, httpclient.Options{
		Timeout:    f.config.HTTPTimeout,
		MaxRetries: f.config.HTTPRetries,
		RetryDelay: 2 * time.Second,
	})
}

func (f *Factory) createFileDownloader() interfaces.FileDownloader {
	return downloader.New(f.httpClient, f.logger)
}

func (f *Factory) createArchiveExtractor() interfaces.ArchiveExtractor {
	return extractor.New(f.logger)
}

func (f *Factory) createCommandRunner() interfaces.CommandRunner {
	return command.NewRunner(f.logger)
}

// Custom tool implementations

type customDownloadableTool struct {
	*tools.DownloadableTool
	url string
}

func (t *customDownloadableTool) GetDownloadURL() (string, error) {
	return t.url, nil
}

type goInstallTool struct {
	*tools.BaseTool
	packageName   string
	commandRunner interfaces.CommandRunner
}

func (t *goInstallTool) Install(ctx context.Context) error {
	if t.IsInstalled() {
		version, _ := t.GetInstalledVersion()
		t.Logger().Info("%s is already installed: %s", t.Name(), version)
		return nil
	}

	t.Logger().Info("Installing %s %s...", t.Name(), t.Version())

	installCmd := fmt.Sprintf("%s@%s", t.packageName, t.Version())
	if err := t.commandRunner.Run(ctx, "go", "install", installCmd); err != nil {
		return fmt.Errorf("failed to install %s: %w", t.Name(), err)
	}

	t.Logger().Success("%s %s installed successfully", t.Name(), t.Version())
	return nil
}
