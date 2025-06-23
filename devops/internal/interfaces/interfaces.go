// Package interfaces defines all interfaces for the devops module
// This promotes dependency injection, testability, and SOLID principles
package interfaces

import (
	"context"
	"io"
	"net/http"
	"time"
)

// Logger interface for logging operations
type Logger interface {
	Info(format string, args ...interface{})
	Success(format string, args ...interface{})
	Warning(format string, args ...interface{})
	Error(format string, args ...interface{})
	Debug(format string, args ...interface{})
}

// HTTPClient interface for HTTP operations
type HTTPClient interface {
	Get(ctx context.Context, url string) (*http.Response, error)
	Do(req *http.Request) (*http.Response, error)
	WaitForHTTP(ctx context.Context, url string, timeout time.Duration) error
	WaitForPort(ctx context.Context, host string, port int, timeout time.Duration) error
	CheckHealth(ctx context.Context, url string, expectedStatus int) error
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

// InstallOptions provides options for tool installation
type InstallOptions struct {
	// Parallel enables parallel installation
	Parallel bool
	// MaxConcurrency limits the number of parallel installations
	MaxConcurrency int
	// SkipValidation skips post-installation validation
	SkipValidation bool
	// Force forces reinstallation even if already installed
	Force bool
}

// ToolRegistry interface for managing multiple tools
type ToolRegistry interface {
	Register(tool ToolInstaller) error
	Get(name string) (ToolInstaller, error)
	List() []ToolInstaller
	Install(ctx context.Context, name string) error
	InstallAll(ctx context.Context) error
	InstallAllWithOptions(ctx context.Context, opts InstallOptions) error
	ValidateAll() error
}

// DockerClient interface for Docker operations
type DockerClient interface {
	IsRunning(ctx context.Context) bool
	ListContainers(ctx context.Context, all bool) ([]ContainerInfo, error)
	PullImage(ctx context.Context, image string) error
	BuildImage(ctx context.Context, path string, tag string, buildArgs map[string]string) error
	RunContainer(ctx context.Context, config ContainerConfig) (string, error)
	StopContainer(ctx context.Context, containerID string, timeout time.Duration) error
	RemoveContainer(ctx context.Context, containerID string, force bool) error
	GetContainerLogs(ctx context.Context, containerID string, follow bool) (io.ReadCloser, error)
	WaitForContainer(ctx context.Context, containerID string) (int, error)
	InspectContainer(ctx context.Context, containerID string) (*ContainerInspect, error)
}

// KubernetesClient interface for Kubernetes operations
type KubernetesClient interface {
	ContextExists(contextName string) (bool, error)
	GetCurrentContext() (string, error)
	WaitForDeployment(ctx context.Context, name, namespace string, timeout time.Duration) error
	WaitForService(ctx context.Context, name, namespace string, timeout time.Duration) error
	GetPods(ctx context.Context, namespace string, labelSelector string) ([]PodInfo, error)
	GetServices(ctx context.Context, namespace string) ([]ServiceInfo, error)
	ApplyManifest(ctx context.Context, manifest []byte) error
	DeleteManifest(ctx context.Context, manifest []byte) error
	GetLogs(ctx context.Context, namespace, podName, containerName string, follow bool) (io.ReadCloser, error)
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

// TaskDocGenerator interface for generating task documentation
type TaskDocGenerator interface {
	Generate(ctx context.Context, rootDir string) (*TaskDocumentation, error)
	GenerateFlow(ctx context.Context, rootDir string) (string, error)
	OutputFormat(doc *TaskDocumentation, format string) ([]byte, error)
}

// ConfigManager interface for configuration management
type ConfigManager interface {
	Load(path string) error
	Get(key string) interface{}
	GetString(key string) string
	GetInt(key string) int
	GetBool(key string) bool
	GetStringSlice(key string) []string
	Set(key string, value interface{})
	Save(path string) error
}

// CommandRunner interface for running external commands
type CommandRunner interface {
	Run(ctx context.Context, name string, args ...string) error
	RunWithOutput(ctx context.Context, name string, args ...string) (string, error)
	RunWithEnv(ctx context.Context, env []string, name string, args ...string) error
}

// TaskExecutor interface for executing Taskfile tasks
type TaskExecutor interface {
	ExecuteTask(ctx context.Context, taskName string, vars map[string]string) error
	LoadTaskfile(taskfilePath string) error
	ValidateDirectory(dir string) error
	ListTasks() ([]TaskInfo, error)
}

// TaskInfo holds information about a task
type TaskInfo struct {
	Name        string
	Description string
	Summary     string
	Vars        map[string]string
	Deps        []string
}

// Data structures used by interfaces

// ContainerInfo holds container information
type ContainerInfo struct {
	ID      string
	Name    string
	Image   string
	Status  string
	State   string
	Health  string
	Created time.Time
	Labels  map[string]string
	Ports   []PortMapping
}

// ContainerConfig holds container configuration
type ContainerConfig struct {
	Image       string
	Name        string
	Cmd         []string
	Env         []string
	Ports       []PortMapping
	Volumes     []VolumeMount
	WorkingDir  string
	Labels      map[string]string
	AutoRemove  bool
	NetworkMode string
}

// ContainerInspect holds detailed container information
type ContainerInspect struct {
	ID              string
	State           ContainerState
	Config          ContainerConfig
	NetworkSettings NetworkSettings
}

// ContainerState holds container state information
type ContainerState struct {
	Status     string
	Running    bool
	Paused     bool
	Restarting bool
	OOMKilled  bool
	Dead       bool
	Pid        int
	ExitCode   int
	Error      string
	StartedAt  time.Time
	FinishedAt time.Time
}

// NetworkSettings holds container network settings
type NetworkSettings struct {
	IPAddress  string
	MacAddress string
	Gateway    string
	Bridge     string
	Ports      map[string][]PortBinding
}

// PortMapping defines port mapping
type PortMapping struct {
	Host      string
	Container string
	Protocol  string
}

// PortBinding defines port binding
type PortBinding struct {
	HostIP   string
	HostPort string
}

// VolumeMount defines volume mount
type VolumeMount struct {
	Source   string
	Target   string
	ReadOnly bool
}

// PodInfo holds pod information
type PodInfo struct {
	Name      string
	Namespace string
	Status    string
	Ready     bool
	IP        string
	Node      string
	Labels    map[string]string
}

// ServiceInfo holds service information
type ServiceInfo struct {
	Name      string
	Namespace string
	Type      string
	ClusterIP string
	Ports     []ServicePort
	Selector  map[string]string
}

// ServicePort defines service port
type ServicePort struct {
	Name       string
	Protocol   string
	Port       int32
	TargetPort int32
	NodePort   int32
}

// TaskDocumentation holds task documentation
type TaskDocumentation struct {
	Taskfiles []TaskfileDoc
	Flow      string
	Generated time.Time
}

// TaskfileDoc holds documentation for a single taskfile
type TaskfileDoc struct {
	Path        string
	Version     string
	Tasks       []TaskDoc
	Includes    map[string]string
	Variables   map[string]interface{}
	Environment map[string]string
}

// TaskDoc holds documentation for a single task
type TaskDoc struct {
	Name          string
	Description   string
	Summary       string
	Commands      []string
	Dependencies  []string
	Sources       []string
	Generates     []string
	Status        []string
	Preconditions []string
	Dir           string
	Vars          map[string]interface{}
	Env           map[string]string
	Silent        bool
	Method        string
}
