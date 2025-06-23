// Package osutil provides OS utility functions
package osutil

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
)

// OSUtil implements the interfaces.OSUtil interface
type OSUtil struct{}

// New creates a new OSUtil instance
func New() interfaces.OSUtil {
	return &OSUtil{}
}

// VerifyCommands checks if commands are available
func (o *OSUtil) VerifyCommands(commands []string) ([]string, error) {
	var missing []string

	for _, cmd := range commands {
		if !o.IsCommandAvailable(cmd) {
			missing = append(missing, cmd)
		}
	}

	if len(missing) > 0 {
		return missing, fmt.Errorf("missing required commands: %v", missing)
	}

	return nil, nil
}

// IsCommandAvailable checks if a command is available in PATH
func (o *OSUtil) IsCommandAvailable(command string) bool {
	_, err := exec.LookPath(command)
	return err == nil
}

// GetOS returns the operating system name
func (o *OSUtil) GetOS() string {
	return runtime.GOOS
}

// GetArch returns the architecture
func (o *OSUtil) GetArch() string {
	arch := runtime.GOARCH
	// Normalize architecture names
	switch arch {
	case "amd64":
		return "amd64"
	case "arm64":
		return "arm64"
	case "386":
		return "386"
	default:
		return arch
	}
}

// GetHomeDir returns the user's home directory
func (o *OSUtil) GetHomeDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return home
}

// CreateTempDir creates a temporary directory
func (o *OSUtil) CreateTempDir(prefix string) (string, error) {
	return os.MkdirTemp("", prefix+"-*")
}

// RemoveTempDir removes a temporary directory
func (o *OSUtil) RemoveTempDir(path string) error {
	return os.RemoveAll(path)
}

// FileExists checks if a file exists
func FileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// IsDirectory checks if a path is a directory
func IsDirectory(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return info.IsDir()
}

// EnsureDir ensures a directory exists
func EnsureDir(path string) error {
	return os.MkdirAll(path, 0755)
}

// GetExecutablePath returns the path of the current executable
func GetExecutablePath() (string, error) {
	return os.Executable()
}

// GetWorkingDir returns the current working directory
func GetWorkingDir() (string, error) {
	return os.Getwd()
}

// ExpandPath expands ~ in paths
func ExpandPath(path string) string {
	if len(path) > 0 && path[0] == '~' {
		home, err := os.UserHomeDir()
		if err == nil {
			path = filepath.Join(home, path[1:])
		}
	}
	return path
}
